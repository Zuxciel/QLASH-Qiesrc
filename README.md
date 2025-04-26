# QLASH: Quantum-Lattice Advanced Secure Hashing

<div align="center">
  <a href="https://github.com/yourusername/qlash"><img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version"></a>
  <a href="https://github.com/yourusername/qlash/releases"><img src="https://img.shields.io/badge/build-20250424-green.svg" alt="Build"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Unlicense-orange.svg" alt="License"></a>
  <br>
  <em>A novel quantum-resistant hash function combining lattice-based cryptography with chaotic systems for enhanced security</em>
</div>

## 🌟 Features

- **Quantum Resistance** - Designed with lattice-based cryptography primitives to withstand quantum computing attacks
- **Dynamic Parameters** - Automatically adjusts internal parameters based on message properties for optimal security
- **Enhanced Diffusion** - Advanced chaotic mixing provides strong avalanche effect
- **High Performance** - SIMD acceleration and multi-threading for optimal speed
- **FFT Integration** - Uses Fast Fourier Transform for frequency-domain mixing operations
- **Flexible Output Size** - Configurable hash output length to meet various security requirements

## 🚀 Performance

QLASH is optimized for performance with several acceleration techniques:

- **Multi-threading** for parallel processing of large inputs
- **SIMD instructions** (AVX/AVX2/AVX512) automatically detected and utilized when available
- **Cache-friendly** algorithms and memory access patterns
- **Optimized permutation and diffusion** operations for minimal computational overhead

## 📊 Benchmark

```
===== QLASH Benchmark Summary (256-bit Output) =====
Total time: 23319.3 ms
Bytes processed: 3417999 bytes (3.25966 MB)
Files processed: 1
Throughput: 0.14 MB/s

--- Time breakdown ---
Preprocessing: 954.76 ms (4.09%)
Lattice operations: 21052.67 ms (90.28%)
Avalanche transform: 576.80 ms (2.47%)
S-box operations: 198.20 ms (0.85%)
Final squeeze: 80.72 ms (0.35%)
=================================

===== QLASH Benchmark Summary (384-bit Output) =====
Total time: 22957.63 ms
Bytes processed: 3417999 bytes (3.26 MB)
Files processed: 1
Throughput: 0.14 MB/s

--- Time breakdown ---
Preprocessing: 895.59 ms (3.90%)
Lattice operations: 20803.15 ms (90.62%)
Avalanche transform: 542.14 ms (2.36%)
S-box operations: 196.81 ms (0.86%)
Final squeeze: 81.20 ms (0.35%)
=================================

===== QLASH Benchmark Summary (512-bit Output) =====
Total time: 23068.92 ms
Bytes processed: 3417999 bytes (3.26 MB)
Files processed: 1
Throughput: 0.14 MB/s

--- Time breakdown ---
Preprocessing: 903.32 ms (3.92%)
Lattice operations: 20880.63 ms (90.51%)
Avalanche transform: 563.54 ms (2.44%)
S-box operations: 196.52 ms (0.85%)
Final squeeze: 82.67 ms (0.36%)
=================================
```

## 📦 Installation

### Prerequisites

- C++17 compatible compiler
- FFTW3 library
- CMake 3.10 or higher

### Building from Source

```bash
git clone https://github.com/yourusername/qlash.git
cd qlash
mkdir build && cd build
cmake ..
make
```

## 💻 Usage

### Basic Usage

```cpp
#include "qlash.h"
#include <string>
#include <iostream>

int main() {
    // Initialize QLASH with default 512-bit output
    qlash::QLASH hasher;
    
    // Hash a string
    std::string input = "Hello, QLASH!";
    std::string hash_result = hasher.hash_string(input);
    
    std::cout << "Hash of '" << input << "': " << hash_result << std::endl;
    
    return 0;
}
```

### Advanced Usage

```cpp
// Create a QLASH instance with custom parameters
// 256-bit output size, 8 threads, enable benchmarking
qlash::QLASH hasher(256, 8, true);

// Hash a file
std::string file_hash = hasher.hash_file("document.pdf");
std::cout << "File hash: " << file_hash << std::endl;

// Get benchmark data
const auto& benchmark = hasher.get_benchmark_data();
std::cout << "Throughput: " << benchmark.get_throughput_mbps() << " MB/s" << std::endl;
```

## 🔧 API Reference

### Constructor

```cpp
QLASH(int output_size_bits = 512, int threads = 4, bool benchmark = false)
```

- `output_size_bits`: Hash output size in bits (default: 512)
- `threads`: Number of threads to use for parallel processing (default: 4)
- `benchmark`: Enable collection of performance statistics (default: false)

### Core Methods

| Method | Description |
|--------|-------------|
| `ByteVector hash(const ByteVector& message)` | Hash raw byte data |
| `std::string hash_to_hex(const ByteVector& message)` | Hash raw data and return hex string |
| `std::string hash_string(const std::string& message)` | Hash a string and return hex result |
| `std::string hash_file(const std::string& filename)` | Hash a file and return hex result |
| `const BenchmarkData& get_benchmark_data() const` | Get performance statistics |
| `void reset_benchmark()` | Reset benchmark data |

## 🔍 Technical Details

### Cryptographic Design

QLASH combines several cryptographic primitives to create a robust hash function:

1. **Input Preprocessing**
   - Dynamic padding based on message properties
   - Initial chaotic transformation

2. **Lattice Operations**
   - Dynamic lattice dimension selection
   - Error-based diffusion using Learning with Errors (LWE) concepts
   - FFT-accelerated mixing

3. **Diffusion Phase**
   - Multiple rounds of substitution and permutation
   - Dynamic S-box generation for each round
   - Strong avalanche effect propagation

4. **Squeezing Phase**
   - Sponge-like construction for output generation
   - Final mixing and extraction

### Security Considerations

- **Quantum Resistance**: Based on the hardness of lattice problems, believed to be resistant to quantum algorithms
- **Side-Channel Protection**: Constant-time operations where possible to mitigate timing attacks
- **Avalanche Effect**: Small input changes propagate rapidly throughout the entire state

## 📜 License

This project is licensed under the Unlicense - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📫 Contact

Project Link: [https://github.com/yourusername/qlash](https://github.com/yourusername/qlash)

---

<div align="center">
  <sub>Built with ❤️ for quantum-resistant cryptography</sub>
</div>
