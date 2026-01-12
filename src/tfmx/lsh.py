import numpy as np

from pathlib import Path
from tclogger import logger
from typing import Optional, TYPE_CHECKING

WEIGHTS_DIR = Path(__file__).parent / "weights"

# Try to import torch for GPU acceleration
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

if TYPE_CHECKING:
    if TORCH_AVAILABLE:
        import torch


class LSHConverter:
    """Convert emb-floats to hash-bits with LSH.
    - dims: input embedding floats dimension (1024)
    - bitn: output hash bits num (2048)
    - seed: random seed for reproducibility (42)
    - use_gpu: whether to use GPU acceleration (default: auto-detect)
    - gpu_threshold: minimum batch size to use GPU (default: 1000)
    """

    def __init__(
        self,
        dims: int = 1024,
        bitn: int = 2048,
        seed: int = 1,
        verbose: bool = False,
        use_gpu: Optional[bool] = None,
        gpu_threshold: int = 500,
    ):
        self.dims = dims
        self.bitn = bitn
        self.seed = seed
        self.verbose = verbose
        self.gpu_threshold = gpu_threshold

        # GPU setup
        if use_gpu is None:
            # Auto-detect: use GPU if available
            self.use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
        else:
            self.use_gpu = use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()

        if self.use_gpu:
            self.device = torch.device("cuda")
            if self.verbose:
                logger.okay(
                    f"  Using GPU acceleration (CUDA) for batches >= {self.gpu_threshold}"
                )
        else:
            self.device = None
            if self.verbose and TORCH_AVAILABLE and not torch.cuda.is_available():
                logger.warn(f"  GPU requested but not available, falling back to CPU")

        self.hps_gpu = None  # Will be set to torch.Tensor if GPU is used
        self.init_hyperplanes()

    def init_hyperplanes(self):
        """init random hyper-planes matrix"""
        if self.verbose:
            logger.note(f"> Init LSH HyperPlanes:", end=" ")
            logger.mesg(f"(dims={self.dims}, bitn={self.bitn}, seed={self.seed})")
        hps_name = f"lsh_hps_sd{self.seed}_{self.dims}_{self.bitn}.npy"
        self.hps_path = WEIGHTS_DIR / hps_name
        if self.hps_path.exists():
            self.load_hyperplanes()
        else:
            self.generate_hyperplanes()
            self.save_hyperplanes()

        # Transfer to GPU if enabled
        if self.use_gpu:
            self.hps_gpu = torch.from_numpy(self.hps).to(self.device)

    def generate_hyperplanes(self):
        np.random.seed(self.seed)
        # generate random hyperplanes: (bitn * dims)
        # each row is a random hyperplane normal vector with dims elements
        self.hps = np.random.randn(self.bitn, self.dims).astype(np.float32)
        # normalize hyperplanes
        self.hps = self.hps / np.linalg.norm(self.hps, axis=1, keepdims=True)

    def embs_to_bits(self, embs: np.ndarray) -> np.ndarray:
        """convert float-embs to hash-bits (ndarray)

        Input:
        - embs: with shape (n, dims) or (dims,)
        - n: number of samples/rows

        Output:
        - bits: with shape (n, bitn) or (bitn,)
        """
        # reshape single row
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        # project embs onto hyperplanes: (n, bitn)
        projs = np.dot(embs, self.hps.T)
        # >0 maps to 1, <=0 maps to 0
        bits = (projs > 0).astype(np.uint8)
        if squeeze:
            return bits[0]
        else:
            return bits

    def bits_to_hex(self, bits: np.ndarray) -> str:
        """convert hash bits to hex str.

        Input:
        - bits: with shape (bits,)

        Output:
        - hex_str: hex str with length len(bits)/4
        """
        # pad to 8x
        bitn = len(bits)
        n_bytes = (bitn + 7) // 8
        padded = np.zeros(n_bytes * 8, dtype=np.uint8)
        padded[:bitn] = bits
        # pack bits into bytes
        bytes_arr = np.packbits(padded)
        hex_str = bytes_arr.tobytes().hex()
        return hex_str

    def embs_to_hex_batch(self, embs: np.ndarray) -> list[str]:
        """Convert embeddings to hex strings in batch (vectorized).

        Uses GPU acceleration if available AND batch size is large enough.
        Small batches use CPU to avoid GPU transfer overhead.

        Input:
        - embs: with shape (n, dims), n samples of embeddings

        Output:
        - hex_strs: list of n hex strings
        """
        # Ensure 2D input
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)

        n_samples = embs.shape[0]

        # Use GPU only if batch is large enough to amortize transfer cost
        use_gpu_for_batch = (
            self.use_gpu
            and self.hps_gpu is not None
            and n_samples >= self.gpu_threshold
        )

        if use_gpu_for_batch:
            # GPU path using PyTorch (for large batches)
            with torch.no_grad():
                # Convert to torch tensor and move to GPU
                embs_gpu = torch.from_numpy(embs).to(self.device)

                # Vectorized projection: (n, dims) @ (dims, bitn) -> (n, bitn)
                projs = torch.matmul(embs_gpu, self.hps_gpu.T)

                # >0 maps to 1, <=0 maps to 0: (n, bitn)
                bits_matrix = (projs > 0).to(torch.uint8).cpu().numpy()
        else:
            # CPU path using NumPy (for small batches or when GPU disabled)
            # Vectorized projection: (n, dims) @ (dims, bitn) -> (n, bitn)
            projs = np.dot(embs, self.hps.T)
            # >0 maps to 1, <=0 maps to 0: (n, bitn)
            bits_matrix = (projs > 0).astype(np.uint8)

        # Pad bitn to multiple of 8 for packbits
        n_bytes = (self.bitn + 7) // 8
        padded_bitn = n_bytes * 8

        if padded_bitn > self.bitn:
            # Pad with zeros on the right
            padding = np.zeros((n_samples, padded_bitn - self.bitn), dtype=np.uint8)
            bits_matrix = np.hstack([bits_matrix, padding])

        # Pack bits into bytes: (n, n_bytes)
        bytes_matrix = np.packbits(bits_matrix, axis=1)

        # Convert each row to hex string
        hex_strs = [row.tobytes().hex() for row in bytes_matrix]
        return hex_strs

    def save_hyperplanes(self):
        logger.note(f"> Save LSH HyperPlanes to:", verbose=self.verbose)
        self.hps_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.hps_path, self.hps)
        logger.okay(f"  * {self.hps_path}", verbose=self.verbose)

    def load_hyperplanes(self):
        logger.note(f"> Load LSH HyperPlanes from:", verbose=self.verbose)
        self.hps = np.load(self.hps_path)
        self.bitn, self.dims = self.hps.shape
        logger.file(f"  * {self.hps_path}", verbose=self.verbose)
