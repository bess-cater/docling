from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Type, List, Any, Sequence

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    OcrOptions,
    PaddleOcrOptions,
)
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from pykospacing import Spacing

_log = logging.getLogger(__name__)


class PaddleOcrModel(BaseOcrModel):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: PaddleOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PaddleOcrOptions
        self.scale = 3  # multiplier for 72 dpi == 216 dpi.
        self.reader = None
        self.spacer = Spacing()

        if self.enabled:
            # Import your external Paddle OCR implementation lazily, trying common paths
            PaddleOCR = None  # type: ignore[assignment]
            try: 
                from paddleocr import PaddleOCR
                PaddleOCR = PaddleOCR
            except Exception as err:
               print("This is error: ", err)

            if PaddleOCR is None:
                raise ImportError(
                    "Could not import PaddleOCR. Tried: paddlex.inference.pipelines.ocr.pipeline, paddlex.inference.pipelines.ocr, paddlex_ocr."
                ) from err

            # Choose a single language string when a list is provided
            lang_value = None
            if isinstance(self.options.lang, list) and len(self.options.lang) > 0:
                # Paddle typically expects a single language string like 'ch', 'en', 'korean', etc.
                lang_value = self.options.lang[0]

            # Instantiate external OCR engine with available options
            self.reader = PaddleOCR(
                lang=lang_value,
                # If your Paddle OCR wrapper supports these toggles, forward them. If not, they will be ignored.
                use_doc_orientation_classify=getattr(
                    self.options, "use_doc_orientation_classify", None
                ),
                use_doc_unwarping=getattr(self.options, "use_doc_unwarping", None),
                use_textline_orientation=getattr(
                    self.options, "use_textline_orientation", None
                ),
            )

    def __del__(self):
        # Best-effort cleanup if the external OCR exposes a close/release method
        try:
            if self.reader is not None and hasattr(self.reader, "close"):
                self.reader.close()
        except Exception:
            pass

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page_i, page in enumerate(page_batch):
            print("This is paddle!!!")
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "ocr"):
                    assert self.reader is not None

                    ocr_rects = self.get_ocr_rects(page)

                    all_ocr_cells: List[TextCell] = []
                    for ocr_rect_i, ocr_rect in enumerate(ocr_rects):
                        if ocr_rect.area() == 0:
                            continue

                        high_res_image = page._backend.get_page_image(
                            scale=self.scale, cropbox=ocr_rect
                        )

                        # Run OCR using the external Paddle OCR reader
                        try:
                            import numpy as np  # local import to avoid hard dependency if unused
                            np_img = np.array(high_res_image)
                            result = self.reader.predict(np_img)
                        except Exception as err:
                            _log.error(
                                "Paddle OCR predict failed for doc %s page %s rect %s: %s",
                                conv_res.input.file,
                                page_i,
                                ocr_rect_i,
                                err,
                            )
                            # Fallback: try passing the PIL image directly
                            try:
                                result = self.reader.predict(high_res_image)
                            except Exception:
                                result = []
                        
                        #print(result)
                        res = result[0]
                        res.save_to_json(f"data/output/paddlee_{page_i}") 

                        # Normalize results to TextCell list
                        cells = [
                            TextCell(
                                index=ix,
                                text=self.spacer(text_, ignore='pre2'),
                                #text = text_,
                                orig=text_,
                                from_ocr=True,
                                confidence=confidence_,
                                rect=BoundingRectangle.from_bounding_box(
                                    BoundingBox.from_tuple(
                                        coord=(
                                            (bbox[0] / self.scale) + ocr_rect.l,
                                            (bbox[1] / self.scale) + ocr_rect.t,
                                            (bbox[2] / self.scale) + ocr_rect.l,
                                            (bbox[3] / self.scale) + ocr_rect.t,
                                        ),
                                        origin=CoordOrigin.TOPLEFT,
                                    )
                                ),
                            )
                            for ix, (text_, bbox, confidence_) in enumerate(zip(res["rec_texts"], res["rec_boxes"], res["rec_scores"]))
                            # if line[2] >= self.options.confidence_threshold
                        ]
                        
                        all_ocr_cells.extend(cells)

                    # Post-process the cells
                    self.post_process_cells(all_ocr_cells, page)

                # DEBUG code:
                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                import gc
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()
                yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return PaddleOcrOptions

    # --- Helpers ---
    def _convert_predictions_to_textcells(
        self,
        predictions: Any,
        ocr_rect: BoundingBox,
        image_size: tuple[int, int],
    ) -> List[TextCell]:
        """Convert Paddle OCR predictions into Docling TextCell objects.

        This is resilient to a few common output formats:
          - [{"text"|"transcription": str, "score"|"conf": float, "points"|"bbox"|"polygon": [[x,y],...]}]
          - [((x0,y0),(x1,y1),(x2,y2),(x3,y3)), text, conf]
          - [([x0,y0,x1,y1,x2,y2,x3,y3], text, conf)]
        """
        textcells: List[TextCell] = []

        if predictions is None:
            return textcells

        # Ensure we always iterate a sequence
        if not isinstance(predictions, Sequence):
            predictions = [predictions]

        for idx, item in enumerate(predictions):
            try:
                # Handle PaddleX OCRResult mapping-like objects
                if hasattr(item, "__getitem__"):
                    try:
                        polys = item["rec_polys"]
                        texts = item["rec_texts"]
                        try:
                            scores = item["rec_scores"]  # type: ignore[index]
                        except Exception:
                            scores = [None] * len(texts)
                    except Exception:
                        polys = texts = scores = None

                    if polys is not None and texts is not None:
                        for j, (poly, text) in enumerate(zip(polys, texts)):
                            conf = None
                            if scores is not None and j < len(scores):
                                conf = scores[j]

                            quad = self._to_quad_points(poly)
                            if quad is None:
                                # Fallback: compute from min/max if poly not in expected format
                                try:
                                    xs = [float(p[0]) for p in poly]
                                    ys = [float(p[1]) for p in poly]
                                    left, top = min(xs), min(ys)
                                    right, bottom = max(xs), max(ys)
                                    quad = [(left, top), (right, top), (right, bottom), (left, bottom)]
                                except Exception:
                                    continue

                            xs = [p[0] for p in quad]
                            ys = [p[1] for p in quad]
                            left, top = min(xs), min(ys)
                            right, bottom = max(xs), max(ys)

                            bbox = BoundingBox(
                                l=(left / self.scale) + ocr_rect.l,
                                t=(top / self.scale) + ocr_rect.t,
                                r=(right / self.scale) + ocr_rect.l,
                                b=(bottom / self.scale) + ocr_rect.t,
                                coord_origin=CoordOrigin.TOPLEFT,
                            )

                            textcells.append(
                                TextCell(
                                    index=len(textcells),
                                    text=str(text),
                                    orig=str(text),
                                    from_ocr=True,
                                    confidence=float(conf) if conf is not None else 0.0,
                                    rect=bbox.to_bounding_rectangle(),
                                )
                            )
                        continue  # Done with this OCRResult

                # Generic parsing fallback
                text, confidence, quad = self._parse_prediction_item(item)
                if text is None or quad is None:
                    continue

                xs = [p[0] for p in quad]
                ys = [p[1] for p in quad]
                left, top = min(xs), min(ys)
                right, bottom = max(xs), max(ys)

                bbox = BoundingBox(
                    l=(left / self.scale) + ocr_rect.l,
                    t=(top / self.scale) + ocr_rect.t,
                    r=(right / self.scale) + ocr_rect.l,
                    b=(bottom / self.scale) + ocr_rect.t,
                    coord_origin=CoordOrigin.TOPLEFT,
                )

                textcells.append(
                    TextCell(
                        index=len(textcells),
                        text=text,
                        orig=text,
                        from_ocr=True,
                        confidence=float(confidence) if confidence is not None else 0.0,
                        rect=bbox.to_bounding_rectangle(),
                    )
                )
            except Exception as err:
                _log.debug("Failed to parse OCR prediction item %s: %s", type(item), err)

        return textcells

    def _parse_prediction_item(self, item: Any):
        """Return (text, confidence, quad) where quad is a 4-point list [(x,y),...]."""
        # Dict-like formats
        if isinstance(item, dict):
            text = item.get("text") or item.get("transcription")
            conf = item.get("score") or item.get("conf") or item.get("confidence")
            pts = item.get("points") or item.get("bbox") or item.get("polygon")
            if text and pts:
                quad = self._to_quad_points(pts)
                return text, conf, quad

        # Tuple/list formats like [poly, text, conf]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            poly = item[0]
            text = item[1]
            conf = item[2] if len(item) > 2 else None
            quad = self._to_quad_points(poly)
            return text, conf, quad

        return None, None, None

    def _to_quad_points(self, pts: Any) -> Optional[List[tuple[float, float]]]:
        """Normalize different polygon formats to a 4-point quad."""
        # Already a list of 4 point pairs
        if (
            isinstance(pts, (list, tuple))
            and len(pts) >= 4
            and isinstance(pts[0], (list, tuple))
        ):
            quad = [(float(x), float(y)) for x, y in pts[:4]]
            return quad

        # Flat list of 8 values
        if isinstance(pts, (list, tuple)) and len(pts) == 8:
            quad = [(float(pts[i]), float(pts[i + 1])) for i in range(0, 8, 2)]
            return quad

        return None
