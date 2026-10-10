"""Pure-Python SafeTensors binary serialization and deserialization engine.

Complies with the official SafeTensors format specification:
- 8-byte unsigned 64-bit little-endian integer indicating header length N
- UTF-8 encoded JSON header mapping tensor names to dtype, shape, and byte offsets
- Contiguous binary data buffer containing little-endian float32 values
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


class SafeTensorsIO:
    """Zero-dependency SafeTensors binary format reader and writer."""

    @staticmethod
    def save_file(
        tensors: Dict[str, Union[List[List[float]], List[float], float, int]],
        filepath: Union[str, Path],
        metadata: Dict[str, str] | None = None,
    ) -> Path:
        """Serialize numeric tensors and metadata into a binary SafeTensors file.

        Args:
            tensors: Mapping of tensor identifiers to nested float lists or scalars.
            filepath: Destination file path.
            metadata: Optional string key-value annotations.

        Returns:
            Resolved Path of the saved file.
        """
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data_buffer = bytearray()
        header: Dict[str, Any] = {}

        if metadata:
            header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

        for name, tensor_data in tensors.items():
            if isinstance(tensor_data, (int, float)):
                flat_vals = [float(tensor_data)]
                shape = [1]
            elif (
                isinstance(tensor_data, list)
                and len(tensor_data) > 0
                and isinstance(tensor_data[0], list)
            ):
                rows = len(tensor_data)
                cols = len(tensor_data[0])
                shape = [rows, cols]
                flat_vals = [float(val) for row in tensor_data for val in row]
            elif isinstance(tensor_data, list):
                shape = [len(tensor_data)]
                flat_vals = [float(val) for val in tensor_data]
            else:
                flat_vals = [float(tensor_data)]
                shape = [1]

            start_offset = len(data_buffer)
            packed_bytes = struct.pack(f"<{len(flat_vals)}f", *flat_vals)
            data_buffer.extend(packed_bytes)
            end_offset = len(data_buffer)

            header[name] = {
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start_offset, end_offset],
            }

        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_len = len(header_json)

        # 8-byte uint64 little-endian length prefix
        prefix = struct.pack("<Q", header_len)

        out_path.write_bytes(prefix + header_json + bytes(data_buffer))
        return out_path

    @staticmethod
    def load_file(filepath: Union[str, Path]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Deserialize a SafeTensors binary file into Python data structures.

        Args:
            filepath: Path to the .safetensors file.

        Returns:
            Tuple of (tensors_dict, metadata_dict).
        """
        in_path = Path(filepath)
        raw_bytes = in_path.read_bytes()

        if len(raw_bytes) < 8:
            raise ValueError(f"File {in_path} is too small to be a valid SafeTensors file.")

        header_len = struct.unpack("<Q", raw_bytes[:8])[0]
        header_end = 8 + header_len

        if len(raw_bytes) < header_end:
            raise ValueError(f"SafeTensors header declared {header_len} bytes, but file truncated.")

        header_raw = raw_bytes[8:header_end].decode("utf-8")
        header = json.loads(header_raw)
        data_buffer = raw_bytes[header_end:]

        metadata: Dict[str, str] = header.pop("__metadata__", {})
        tensors: Dict[str, Any] = {}

        for tensor_name, tensor_spec in header.items():
            start_off, end_off = tensor_spec["data_offsets"]
            shape: List[int] = tensor_spec.get("shape", [])
            byte_count = end_off - start_off
            elem_count = byte_count // 4

            slice_bytes = data_buffer[start_off:end_off]
            unpacked = list(struct.unpack(f"<{elem_count}f", slice_bytes))

            if len(shape) == 2:
                rows, cols = shape
                tensors[tensor_name] = [unpacked[r * cols : (r + 1) * cols] for r in range(rows)]
            elif len(shape) == 1 and shape[0] == 1:
                tensors[tensor_name] = unpacked[0]
            else:
                tensors[tensor_name] = unpacked

        return tensors, metadata
