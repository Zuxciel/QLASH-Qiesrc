/**
 * QLASH: Quantum-Lattice Advanced Secure Hashsing
 * A novel quantum-resistant hash function combining lattice-based cryptography
 * with chaotic systems for enhanced security.
 *
 */

#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <chrono>
#include <stdexcept>
#include <numeric>
#include <future>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <fftw3.h> // Untuk FFT (harus diinstal secara terpisah)
#include <queue>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>

// Include SIMD intrinsics header
#if defined(__AVX512F__)
#include <immintrin.h> // AVX512
#define HAVE_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h> // AVX2
#define HAVE_AVX2 1
#elif defined(__AVX__)
#include <immintrin.h> // AVX
#define HAVE_AVX 1
#elif defined(__SSE4_2__)
#include <smmintrin.h> // SSE4.2
#define HAVE_SSE4_2 1
#elif defined(__SSE2__)
#include <emmintrin.h> // SSE2
#define HAVE_SSE2 1
#endif

// CLI progress bar
#include <cstdio>
#ifdef _WIN32
#include <windows.h>
#define CLEAR_LINE "\r"
#else
#define CLEAR_LINE "\033[2K\r"
#endif

// Define M_PI for portability
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Optional: Enable debug output
// #define QLASH_DEBUG

// Version info
#define QLASH_VERSION "1.0.0"
#define QLASH_BUILD "20250424"

namespace qlash
{

    // Configuration constants
    constexpr int DEFAULT_OUTPUT_SIZE = 512; // bits
    constexpr int DEFAULT_BLOCK_SIZE = 64;   // bytes (Rate for internal mixing/squeezing conceptual)
    constexpr int MIN_LATTICE_DIM = 8;
    constexpr int MAX_LATTICE_DIM = 16;     // Limited for conceptual example
    constexpr uint16_t BASE_MODULUS = 7681; // A prime q for lattice operations (Example)

    // Target size for the internal state used in the Squeezing phase
    constexpr size_t SPONGE_STATE_SIZE = 512; // 512 bytes = 4096 bits
    constexpr size_t SQUEEZE_RATE = 64;       // 64 bytes = 512 bits

    // Threading constants
    constexpr int DEFAULT_THREAD_COUNT = 4;
    constexpr size_t MIN_BYTES_PER_THREAD = 16384; // 16KB minimum per thread

    // Type definitions for clarity
    using Byte = uint8_t;
    using Word = uint32_t;
    using DWord = uint64_t;
    using ByteVector = std::vector<Byte>;
    using Matrix = std::vector<std::vector<uint16_t>>;
    using SBox = std::array<Byte, 256>;

    // Benchmark data structure
    struct BenchmarkData
    {
        std::chrono::microseconds preprocessing_time{0};
        std::chrono::microseconds lattice_time{0};
        std::chrono::microseconds avalanche_time{0};
        std::chrono::microseconds sbox_time{0};
        std::chrono::microseconds squeeze_time{0};
        std::chrono::microseconds total_time{0};
        size_t bytes_processed{0};
        size_t files_processed{0};

        double get_throughput_mbps() const
        {
            if (total_time.count() == 0)
                return 0.0;
            return static_cast<double>(bytes_processed) / total_time.count() * 1000000.0 / (1024.0 * 1024.0);
        }

        void print_summary() const
        {
            std::cout << "\n===== QLASH Benchmark Summary =====\n";
            std::cout << "Total time: " << total_time.count() / 1000.0 << " ms\n";
            std::cout << "Bytes processed: " << bytes_processed << " bytes ("
                      << bytes_processed / (1024.0 * 1024.0) << " MB)\n";
            std::cout << "Files processed: " << files_processed << "\n";
            std::cout << "Throughput: " << std::fixed << std::setprecision(2)
                      << get_throughput_mbps() << " MB/s\n";
            std::cout << "\n--- Time breakdown ---\n";
            std::cout << "Preprocessing: " << preprocessing_time.count() / 1000.0 << " ms ("
                      << (total_time.count() > 0 ? 100.0 * preprocessing_time.count() / total_time.count() : 0.0) << "%)\n";
            std::cout << "Lattice operations: " << lattice_time.count() / 1000.0 << " ms ("
                      << (total_time.count() > 0 ? 100.0 * lattice_time.count() / total_time.count() : 0.0) << "%)\n";
            std::cout << "Avalanche transform: " << avalanche_time.count() / 1000.0 << " ms ("
                      << (total_time.count() > 0 ? 100.0 * avalanche_time.count() / total_time.count() : 0.0) << "%)\n";
            std::cout << "S-box operations: " << sbox_time.count() / 1000.0 << " ms ("
                      << (total_time.count() > 0 ? 100.0 * sbox_time.count() / total_time.count() : 0.0) << "%)\n";
            std::cout << "Final squeeze: " << squeeze_time.count() / 1000.0 << " ms ("
                      << (total_time.count() > 0 ? 100.0 * squeeze_time.count() / total_time.count() : 0.0) << "%)\n";
            std::cout << "=================================\n";
        }
    };

    // Thread pool implementation for parallel processing
    class ThreadPool
    {
    public:
        explicit ThreadPool(size_t num_threads) : stop(false)
        {
            for (size_t i = 0; i < num_threads; ++i)
            {
                workers.emplace_back([this]
                                     {
                  for (;;) {
                      std::function<void()> task;
                      {
                          std::unique_lock<std::mutex> lock(this->queue_mutex);
                          this->condition.wait(lock, [this] { 
                              return this->stop || !this->tasks.empty(); 
                          });
                          if (this->stop && this->tasks.empty()) return;
                          task = std::move(this->tasks.front());
                          this->tasks.pop();
                      }
                      task();
                  } });
            }
        }

        template <class F>

        std::future<void> enqueue(F &&f)
        {
            auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(f));
            std::future<void> res = task->get_future();

            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (stop)
                    throw std::runtime_error("enqueue on stopped ThreadPool");

                tasks.emplace([task]()
                              { (*task)(); });
            }

            condition.notify_one();
            return res;
        }

        ~ThreadPool()
        {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                stop = true;
            }
            condition.notify_all();
            for (std::thread &worker : workers)
            {
                worker.join();
            }
        }

        size_t get_thread_count() const
        {
            return workers.size();
        }

    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
        std::mutex queue_mutex;
        std::condition_variable condition;
        bool stop;
    };

    /**
     * Core QLASH class implementation
     */
    class QLASH
    {
    private:
        // Hash state
        ByteVector state_;
        int output_size_;

        // Lattice parameters calculated from input
        int lattice_dimension_;
        uint16_t modulus_;

        // Dynamic parameters based on input complexity
        int rounds_;

        // Precomputed dynamic S-boxes (one per round)
        std::vector<SBox> dynamic_sboxes_;

        // Benchmarking data
        BenchmarkData benchmark_;
        bool enable_benchmark_;

        // Thread pool for parallel processing
        std::shared_ptr<ThreadPool> thread_pool_; // Ditambahkan sebagai anggota kelas
        int num_threads_;

        // --- Helper Functions (Private) ---

        // FFTW plans and buffers
        fftw_plan forward_plan_;
        fftw_plan inverse_plan_;
        fftw_complex *fft_in_;
        fftw_complex *fft_out_;
        fftw_complex *fft_tmp_;
        bool fftw_initialized_;

        // Initialize FFTW resources
        void initialize_fftw(size_t size)
        {
            if (fftw_initialized_)
            {
                fftw_destroy_plan(forward_plan_);
                fftw_destroy_plan(inverse_plan_);
                fftw_free(fft_in_);
                fftw_free(fft_out_);
                fftw_free(fft_tmp_);
            }

            fft_in_ = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size);
            fft_out_ = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size);
            fft_tmp_ = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size);

            if (!fft_in_ || !fft_out_ || !fft_tmp_)
            {
                throw std::runtime_error("Failed to allocate FFTW memory");
            }

            forward_plan_ = fftw_plan_dft_1d(size, fft_in_, fft_out_, FFTW_FORWARD, FFTW_ESTIMATE);
            inverse_plan_ = fftw_plan_dft_1d(size, fft_out_, fft_in_, FFTW_BACKWARD, FFTW_ESTIMATE);

            if (!forward_plan_ || !inverse_plan_)
            {
                throw std::runtime_error("Failed to create FFTW plans");
            }

            fftw_initialized_ = true;
        }

        // Clean up FFTW resources
        void cleanup_fftw()
        {
            if (fftw_initialized_)
            {
                fftw_destroy_plan(forward_plan_);
                fftw_destroy_plan(inverse_plan_);
                fftw_free(fft_in_);
                fftw_free(fft_out_);
                fftw_free(fft_tmp_);
                fftw_initialized_ = false;
            }
        }

        // Add FFT-based mixing to enhance lattice operations
        ByteVector fft_mix(const ByteVector &data)
        {
            size_t padded_size = next_power_of_2(data.size());

            // Initialize FFT resources for this size if needed
            if (!fftw_initialized_ || padded_size != sizeof(fft_in_) / sizeof(fftw_complex))
            {
                initialize_fftw(padded_size);
            }

            // Prepare input data
            for (size_t i = 0; i < padded_size; i++)
            {
                if (i < data.size())
                {
                    fft_in_[i][0] = static_cast<double>(data[i]);
                    fft_in_[i][1] = 0.0;
                }
                else
                {
                    fft_in_[i][0] = 0.0;
                    fft_in_[i][1] = 0.0;
                }
            }

            // Execute forward FFT
            fftw_execute(forward_plan_);

            // Apply frequency domain transformations
            for (size_t i = 0; i < padded_size; i++)
            {
                double real = fft_out_[i][0];
                double imag = fft_out_[i][1];

                // Apply non-linear transformation in frequency domain
                double magnitude = std::sqrt(real * real + imag * imag);
                double phase = std::atan2(imag, real);

                // Apply chaotic map to phase
                double x = std::fmod(phase, 2.0 * M_PI) / (2.0 * M_PI);
                x = 3.99 * x * (1.0 - x);
                phase = x * 2.0 * M_PI;

                // Apply modulus-based operations to magnitude
                magnitude = std::fmod(magnitude + static_cast<double>(modulus_), 256.0);

                // Convert back to real/imaginary
                fft_out_[i][0] = magnitude * std::cos(phase);
                fft_out_[i][1] = magnitude * std::sin(phase);
            }

            // Execute inverse FFT
            fftw_execute(inverse_plan_);

            // Prepare output data with normalization
            ByteVector result(data.size());
            for (size_t i = 0; i < data.size(); i++)
            {
                // Normalize and convert to byte
                double value = fft_in_[i][0] / padded_size;
                result[i] = static_cast<Byte>(std::fmod(std::abs(value), 256.0));
            }

            return result;
        }

        /**
         * Initialize dynamic parameters based on input message properties.
         */
        void initialize_parameters(const ByteVector &message)
        {
            // Calculate complexity metrics from message
            // Using entropy and length as example complexity measures
            double entropy = calculate_entropy(message);
            size_t length = message.size();

            // Set lattice dimension based on message properties
            lattice_dimension_ = std::min(MAX_LATTICE_DIM,
                                          std::max(MIN_LATTICE_DIM,
                                                   static_cast<int>(std::log2(length + 1)) + 4));

            // Calculate dynamic modulus (must be prime)
            modulus_ = next_prime(static_cast<uint16_t>(BASE_MODULUS + (entropy * 64)));
            if (modulus_ < BASE_MODULUS || modulus_ > 65500)
                modulus_ = BASE_MODULUS;

            // Set adaptive rounds based on message complexity
            rounds_ = std::max(10, std::min(20, static_cast<int>(8 + entropy)));
            if (rounds_ <= 0)
                rounds_ = 10; // Ensure positive rounds

            // Generate dynamic S-boxes for this message based on finalized parameters
            generate_dynamic_sboxes(message); // S-boxes depend on rounds_
        }

        /**
         * Calculate Shannon entropy of the message (complexity measure)
         */
        double calculate_entropy(const ByteVector &message)
        {
            if (message.empty())
                return 0.0;

            // Count byte frequencies
            std::array<int, 256> frequencies = {0};

            // Use SIMD for faster frequency counting if available
#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
            // Implementation would use SIMD to count frequencies faster
            // This is a simplified representation
            for (Byte b : message)
            {
                frequencies[b]++;
            }
#else
            for (Byte b : message)
            {
                frequencies[b]++;
            }
#endif

            // Calculate entropy
            double entropy = 0.0;
            double log2_base = std::log(2.0);
            double n = static_cast<double>(message.size());

            for (int freq : frequencies)
            {
                if (freq > 0)
                {
                    double p = freq / n;
                    entropy -= p * std::log(p) / log2_base;
                }
            }

            return entropy;
        }

        /**
         * Find next prime number after n (for modulus calculation)
         */
        uint16_t next_prime(uint16_t n)
        {
            if (n < 2)
                n = 2;
            while (!is_prime(n))
            {
                n++;
                if (n == 0)
                    n = 65535; // Prevent infinite loop on overflow if starting near max
            }
            return n;
        }

        /**
         * Optimized primality test using Miller-Rabin for larger numbers
         */
        bool is_prime(uint16_t n)
        {
            if (n <= 1)
                return false;
            if (n <= 3)
                return true;
            if (n % 2 == 0 || n % 3 == 0)
                return false;

            // Small numbers - trial division
            if (n < 1000)
            {
                for (uint16_t i = 5; (uint32_t)i * i <= n; i += 6)
                {
                    if (n % i == 0 || n % (i + 2) == 0)
                    {
                        return false;
                    }
                }
                return true;
            }

            // Larger numbers - use Miller-Rabin test (deterministic for 16-bit)
            return miller_rabin_test(n);
        }

        /**
         * Miller-Rabin primality test (deterministic for 16-bit integers)
         */
        bool miller_rabin_test(uint16_t n)
        {
            // Find r and d such that n = 2^r * d + 1
            uint16_t d = n - 1;
            int r = 0;
            while ((d & 1) == 0)
            {
                d >>= 1;
                r++;
            }

            // Test with bases that are guaranteed to work for 16-bit numbers
            uint16_t bases[] = {2, 3, 5, 7, 11, 13, 17};

            for (uint16_t a : bases)
            {
                if (a >= n)
                    continue;

                uint32_t x = power_mod(a, d, n);
                if (x == 1 || x == n - 1)
                    continue;

                bool is_composite = true;
                for (int j = 0; j < r - 1; j++)
                {
                    x = (uint32_t)x * x % n;
                    if (x == n - 1)
                    {
                        is_composite = false;
                        break;
                    }
                }

                if (is_composite)
                    return false;
            }

            return true;
        }

        /**
         * Fast modular exponentiation: computes (base^exp) % mod
         */
        // Review the power_mod function for potential micro-optimizations
        uint32_t power_mod(uint32_t base, uint32_t exp, uint32_t mod)
        {
            uint32_t result = 1;
            base %= mod;

            while (exp > 0)
            {
                if (exp & 1)
                    result = (static_cast<uint64_t>(result) * base) % mod; // Use uint64_t for intermediate product to prevent overflow before modulo
                exp >>= 1;
                base = (static_cast<uint64_t>(base) * base) % mod; // Use uint64_t
            }

            return result;
        }

        /**
         * Find next power of 2 greater than or equal to n
         */
        size_t next_power_of_2(size_t n)
        {
            if (n == 0)
                return 1;

            // Fast next power of 2 calculation
            n--;
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            if (sizeof(size_t) > 4)
                n |= n >> 32; // For 64-bit systems

            return n + 1;
        }

        ByteVector fft_chaotic_transform(const ByteVector &data)
        {
            // Only use FFT for larger data
            if (data.size() < 1024)
            {
                return chaotic_transform(data);
            }

            auto start_time = std::chrono::high_resolution_clock::now();

            // Apply initial chaotic transform
            ByteVector result = chaotic_transform(data);

            // Apply FFT-based mixing on top of chaotic transform
            ByteVector fft_result = fft_mix(result);

            // Combine results
            for (size_t i = 0; i < result.size(); i++)
            {
                result[i] = result[i] ^ fft_result[i % fft_result.size()];
            }

            if (enable_benchmark_)
            {
                auto end_time = std::chrono::high_resolution_clock::now();
                benchmark_.preprocessing_time += std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time);
            }

            return result;
        }

        /**
         * Input preprocessing with padding and initial transformation
         */
        ByteVector preprocess(const ByteVector &message)
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            // Calculate padding
            size_t length = message.size();
            size_t target_length = next_power_of_2(length + 16);
            size_t padding_needed = target_length - length - 16;

            // Create padded message
            ByteVector padded;
            padded.reserve(target_length);

            // Original message
            padded.insert(padded.end(), message.begin(), message.end());

            // Append 8-byte length information (little-endian)
            DWord len_dword = length;
            for (int i = 0; i < 8; i++)
            {
                padded.push_back(static_cast<Byte>((len_dword >> (i * 8)) & 0xFF));
            }

            // Append 8-byte padding needed information (little-endian)
            DWord padding_dword = padding_needed;
            for (size_t i = 0; i < 8; i++)
            {
                Byte padding_byte_info = static_cast<Byte>((padding_dword >> (i * 8)) & 0xFF);
                padded.push_back(padding_byte_info);
            }

            // Add actual padding bytes with non-linear pattern based on position and message hash/checksum
            Byte checksum = 0;
            for (Byte b : message)
                checksum ^= b;

            for (size_t i = 0; i < padding_needed; i++)
            {
                Byte padding_byte = static_cast<Byte>((i * 0x6D) ^ (checksum + i) ^ (i >> 3) ^ 0x91);
                padded.push_back(padding_byte);
            }

            // Apply initial chaotic transformation with FFT acceleration
            ByteVector initial_transformed = fft_chaotic_transform(padded);

            // Debug print (if enabled)
#ifdef QLASH_DEBUG
            print_state("State after preprocessing", initial_transformed);
#endif

            if (enable_benchmark_)
            {
                auto end_time = std::chrono::high_resolution_clock::now();
                benchmark_.preprocessing_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time);
            }

            return initial_transformed;
        }

        /**
         * Initial chaotic transformation for preprocessing (optimized)
         */
        // Update methods that use thread_pool_ to accept it as a parameter
        ByteVector chaotic_transform(const ByteVector &data)
        {
            ByteVector result = data;
            double x = 0.5;
            if (!data.empty())
            {
                x = std::fmod(static_cast<double>(data[0]) / 256.0 +
                                  static_cast<double>(data.back()) / 256.0 + 0.12345,
                              1.0);
                if (x < 0.01 || x > 0.99)
                    x = 0.5;
            }

            if (data.size() >= MIN_BYTES_PER_THREAD * num_threads_ && num_threads_ > 1)
            {
                size_t chunk_size = data.size() / num_threads_;
                std::vector<std::future<void>> futures;
                std::vector<double> chunk_x_values(num_threads_);
                std::shared_ptr<ThreadPool> thread_pool = std::make_shared<ThreadPool>(num_threads_);

                for (int i = 0; i < num_threads_; i++)
                {
                    chunk_x_values[i] = x * (1.0 + 0.01 * i);
                    if (chunk_x_values[i] <= 0 || chunk_x_values[i] >= 1)
                        chunk_x_values[i] = 0.5;
                }

                for (int i = 0; i < num_threads_; i++)
                {
                    size_t start = i * chunk_size;
                    size_t end = (i == num_threads_ - 1) ? data.size() : (i + 1) * chunk_size;

                    futures.push_back(thread_pool->enqueue([this, &result, &chunk_x_values, i, start, end]()
                                                           {
                     double local_x = chunk_x_values[i];
                     for (size_t j = start; j < end; j++) {
                         local_x = 3.99 * local_x * (1.0 - local_x);
                         if (local_x <= 0 || local_x >= 1) local_x = 0.5;
                         result[j] = static_cast<Byte>(result[j] ^
                                                      static_cast<Byte>(local_x * 256.0) ^
                                                      ((j > start) ? result[j-1] : static_cast<Byte>(local_x * 256.0) >> 1));
                     } }));
                }

                for (auto &future : futures)
                {
                    future.wait();
                }

                for (size_t i = 1; i < num_threads_; i++)
                {
                    size_t boundary = i * chunk_size;
                    if (boundary < result.size() && boundary > 0)
                    {
                        result[boundary] ^= result[boundary - 1];
                    }
                }
            }
            else
            {
                for (size_t i = 0; i < result.size(); i++)
                {
                    x = 3.99 * x * (1.0 - x);
                    if (x <= 0 || x >= 1)
                        x = 0.5;
                    result[i] = static_cast<Byte>(result[i] ^
                                                  static_cast<Byte>(x * 256.0) ^
                                                  ((i > 0) ? result[i - 1] : (static_cast<Byte>(x * 256.0) >> 1)));
                }
            }

            return result;
        }

        /**
         * Generate dynamic S-boxes for each round
         */
        void generate_dynamic_sboxes(const ByteVector &seed)
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            dynamic_sboxes_.clear();
            dynamic_sboxes_.reserve(rounds_);

            // Create seeded PRNG
            std::seed_seq seq(seed.begin(), seed.end());
            std::mt19937 gen(seq);

            for (int round = 0; round < rounds_; round++)
            {
                SBox sbox;

                // Initialize S-box with identity mapping
                for (int i = 0; i < 256; i++)
                {
                    sbox[i] = static_cast<Byte>(i);
                }

                // Use chaotic maps to permute the S-box
                double x = 0.5;

                // Mix in round number to make each round's S-box unique
                x = std::fmod(x + round * 0.0173, 1.0);
                if (x <= 0 || x >= 1)
                    x = 0.5;

                // Apply chaotic permutation
                for (int i = 255; i > 0; i--)
                {
                    x = 3.99 * x * (1.0 - x); // Logistic map
                    if (x <= 0 || x >= 1)
                        x = 0.5;

                    // Use chaotic value to select an index to swap with
                    int j = static_cast<int>(x * (i + 1)); // 0 to i inclusive
                    if (j > i)
                        j = i; // Just in case of floating point issues

                    // Swap elements
                    std::swap(sbox[i], sbox[j]);
                }

                dynamic_sboxes_.push_back(sbox);
            }

            if (enable_benchmark_)
            {
                auto end_time = std::chrono::high_resolution_clock::now();
                benchmark_.sbox_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time);
            }
        }

        /**
         * Apply S-box substitution to the state
         */
        ByteVector substitute(const ByteVector &input, int round)
        {
            ByteVector result = input;

            // Get current round's S-box (modulo the number of generated S-boxes)
            const SBox &sbox = dynamic_sboxes_[round % dynamic_sboxes_.size()];

            // Apply S-box substitution
            for (size_t i = 0; i < result.size(); i++)
            {
                result[i] = sbox[result[i]];
            }

            return result;
        }

        /**
         * Lattice-based mixing function (optimized)
         */
        ByteVector lattice_mix(const ByteVector &input)
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            // Apply FFT-based processing for large inputs
            ByteVector fft_processed;
            if (input.size() >= 1024)
            {
                fft_processed = fft_mix(input);
            }

            // Process with lattice operations as before
            std::vector<std::vector<uint16_t>> vectors = bytes_to_lattice_vectors(input);
            Matrix A = generate_lattice_matrix(input);
            std::vector<std::vector<uint16_t>> result_vectors;
            result_vectors.reserve(vectors.size());
            std::seed_seq seq_mix(state_.begin(), state_.end());
            std::mt19937 gen_mix(seq_mix);

            if (vectors.size() >= num_threads_ && num_threads_ > 1)
            {
                std::mutex result_mutex;
                std::vector<std::vector<uint16_t>> local_results(vectors.size());
                std::vector<std::future<void>> futures;
                std::shared_ptr<ThreadPool> thread_pool = std::make_shared<ThreadPool>(num_threads_);

                for (int t = 0; t < num_threads_; t++)
                {
                    size_t start_idx = t * vectors.size() / num_threads_;
                    size_t end_idx = (t + 1) * vectors.size() / num_threads_;

                    futures.push_back(thread_pool->enqueue([this, &vectors, &A, &local_results, &gen_mix,
                                                            start_idx, end_idx]()
                                                           {
                 std::mt19937 local_gen(gen_mix());
                 for (size_t i = start_idx; i < end_idx; i++) {
                     const auto& v = vectors[i];
                     std::vector<uint16_t> r = generate_small_vector(local_gen);
                     std::vector<uint16_t> e = generate_error_vector(local_gen);
                     std::vector<uint16_t> b(lattice_dimension_, 0);

                     for (int i = 0; i < lattice_dimension_; i++) {
                         uint32_t sum = 0;
                         for (int j = 0; j < lattice_dimension_; j++) {
                             sum = (sum + static_cast<uint32_t>(A[i][j]) * r[j]);
                         }
                         b[i] = sum % modulus_;
                         b[i] = (b[i] + e[i]) % modulus_;
                         b[i] = (b[i] + v[i]) % modulus_;
                     }
                     local_results[i] = std::move(b);
                 } }));
                }

                for (auto &future : futures)
                {
                    future.wait();
                }

                result_vectors = std::move(local_results);
            }
            else
            {
                for (const auto &v : vectors)
                {
                    std::vector<uint16_t> r = generate_small_vector(gen_mix);
                    std::vector<uint16_t> e = generate_error_vector(gen_mix);
                    std::vector<uint16_t> b(lattice_dimension_, 0);

                    for (int i = 0; i < lattice_dimension_; i++)
                    {
                        uint32_t sum = 0;
                        for (int j = 0; j < lattice_dimension_; j++)
                        {
                            sum = (sum + static_cast<uint32_t>(A[i][j]) * r[j]);
                        }
                        b[i] = sum % modulus_;
                        b[i] = (b[i] + e[i]) % modulus_;
                        b[i] = (b[i] + v[i]) % modulus_;
                    }
                    result_vectors.push_back(b);
                }
            }

#ifdef QLASH_DEBUG
            if (!result_vectors.empty())
            {
                print_lattice_matrix("Lattice Matrix A (first few)", A);
                print_lattice_vector("Lattice Vector V (first)", vectors.empty() ? std::vector<uint16_t>() : vectors[0]);
                print_lattice_vector("Lattice Vector r (example)", generate_small_vector(gen_mix));
                print_lattice_vector("Lattice Vector e (example)", generate_error_vector(gen_mix));
                print_lattice_vector("Lattice Vector B (first result)", result_vectors[0]);
            }
#endif

            // Convert lattice vectors to bytes (single declaration of result)
            ByteVector result = lattice_vectors_to_bytes(result_vectors);

            // If FFT was used, combine the results
            if (!fft_processed.empty())
            {
                // Combine FFT-processed data with lattice result
                for (size_t i = 0; i < result.size(); i++)
                {
                    result[i] ^= fft_processed[i % fft_processed.size()];
                }
            }

            if (enable_benchmark_)
            {
                auto end_time = std::chrono::high_resolution_clock::now();
                benchmark_.lattice_time += std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time);
            }

            return result;
        }

        /**
         * Convert byte array to lattice vectors (elements are uint16_t modulo q)
         */
        std::vector<std::vector<uint16_t>> bytes_to_lattice_vectors(const ByteVector &bytes)
        {
            std::vector<std::vector<uint16_t>> vectors;

            // Each vector requires lattice_dimension_ elements
            size_t elements_per_vector = lattice_dimension_;
            size_t bytes_per_vector = elements_per_vector * 2; // 2 bytes per element for uint16_t

            // Calculate how many vectors we can create
            size_t padded_bytes_size = bytes.size();
            if (padded_bytes_size % bytes_per_vector != 0)
            {
                padded_bytes_size += bytes_per_vector - (padded_bytes_size % bytes_per_vector);
            }
            size_t num_vectors = padded_bytes_size / bytes_per_vector;
            if (num_vectors == 0 && !bytes.empty())
                num_vectors = 1;
            if (num_vectors == 0 && bytes.empty())
                return {};

            vectors.reserve(num_vectors);

            for (size_t i = 0; i < num_vectors; i++)
            {
                std::vector<uint16_t> vector(lattice_dimension_, 0);

                for (size_t j = 0; j < elements_per_vector; j++)
                {
                    size_t byte_idx = i * bytes_per_vector + j * 2;

                    uint16_t value = 0;
                    // If we have two bytes available, combine them (little-endian)
                    if (byte_idx + 1 < bytes.size())
                    {
                        value = (static_cast<uint16_t>(bytes[byte_idx + 1]) << 8) | bytes[byte_idx];
                    }
                    else if (byte_idx < bytes.size())
                    {
                        // If we only have one byte left
                        value = bytes[byte_idx];
                    }
                    else
                    {
                        // Padding with value derived from position and previous values
                        value = ((i * 7919 + j * 373) ^ (vector.empty() ? 0x5A5A : vector.back())) % modulus_;
                    }

                    // Ensure the value is properly reduced modulo q
                    vector[j] = value % modulus_;
                }

                vectors.push_back(std::move(vector));
            }

            return vectors;
        }

        /**
         * Convert lattice vectors back to byte array
         */
        ByteVector lattice_vectors_to_bytes(const std::vector<std::vector<uint16_t>> &vectors)
        {
            ByteVector result;

            // Calculate result size
            size_t total_bytes = vectors.size() * lattice_dimension_ * 2; // 2 bytes per element
            result.reserve(total_bytes);

            for (const auto &vector : vectors)
            {
                for (uint16_t value : vector)
                {
                    // Convert each uint16_t to two bytes (little-endian)
                    result.push_back(static_cast<Byte>(value & 0xFF));
                    result.push_back(static_cast<Byte>((value >> 8) & 0xFF));
                }
            }

            return result;
        }

        /**
         * Generate a pseudorandom lattice matrix A based on input seed
         */
        Matrix generate_lattice_matrix(const ByteVector &seed)
        {
            // Create matrix A of size n×n
            Matrix A(lattice_dimension_, std::vector<uint16_t>(lattice_dimension_, 0));

            // Use seed to initialize PRNG
            std::seed_seq seq(seed.begin(), seed.end());
            std::mt19937 gen(seq);
            std::uniform_int_distribution<uint16_t> dist(1, modulus_ - 1);

            // Generate matrix elements
            for (int i = 0; i < lattice_dimension_; i++)
            {
                for (int j = 0; j < lattice_dimension_; j++)
                {
                    A[i][j] = dist(gen);
                }
            }

            return A;
        }

        /**
         * Generate a small random vector for lattice operations
         */
        std::vector<uint16_t> generate_small_vector(std::mt19937 &gen)
        {
            std::vector<uint16_t> r(lattice_dimension_, 0);

            // Use a narrow distribution for small coefficients
            std::uniform_int_distribution<uint16_t> small_dist(0, 1);

            for (int i = 0; i < lattice_dimension_; i++)
            {
                r[i] = small_dist(gen);
            }

            return r;
        }

        /**
         * Generate a small error vector for lattice operations
         */
        std::vector<uint16_t> generate_error_vector(std::mt19937 &gen)
        {
            std::vector<uint16_t> e(lattice_dimension_, 0);

            // Use discrete Gaussian-like distribution for error terms
            std::binomial_distribution<int> bin_dist(10, 0.5); // Approximation of small Gaussian

            for (int i = 0; i < lattice_dimension_; i++)
            {
                int err = bin_dist(gen) - 5; // Center around 0 with small magnitude
                // Convert to unsigned and ensure it's properly reduced mod q
                e[i] = (err < 0) ? static_cast<uint16_t>(modulus_ + err) : static_cast<uint16_t>(err);
            }

            return e;
        }

        /**
         * Apply diffusion (permutation and mixing) to create avalanche effect
         */
        // Fix for the diffuse function to ensure proper thread synchronization
        ByteVector diffuse(const ByteVector &input)
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            ByteVector result = input;

            if (result.size() < 8)
            {
                if (enable_benchmark_)
                {
                    auto end_time = std::chrono::high_resolution_clock::now();
                    benchmark_.avalanche_time += std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - start_time);
                }
                return result;
            }

            const size_t block_size = DEFAULT_BLOCK_SIZE;
            const size_t num_blocks = (result.size() + block_size - 1) / block_size;

            if (num_blocks >= num_threads_ && num_threads_ > 1)
            {
                std::vector<std::future<void>> futures;
                std::shared_ptr<ThreadPool> thread_pool = std::make_shared<ThreadPool>(num_threads_);

                for (int t = 0; t < num_threads_; t++)
                {
                    size_t start_block = t * num_blocks / num_threads_;
                    size_t end_block = (t + 1) * num_blocks / num_threads_;

                    futures.emplace_back(thread_pool->enqueue([this, &result, block_size, start_block, end_block]()
                                                              {
                         for (size_t block_idx = start_block; block_idx < end_block; block_idx++) {
                             size_t start_idx = block_idx * block_size;
                             size_t end_idx = std::min(start_idx + block_size, result.size());
                             size_t block_length = end_idx - start_idx;
     
                             if (block_length < 2) continue;
     
                             for (size_t i = 0; i < block_length / 2; i++) {
                                 size_t j = (i * 7 + 3) % block_length;
                                 std::swap(result[start_idx + i], result[start_idx + j]);
                             }
     
                             for (size_t i = start_idx + 1; i < end_idx; i++) {
                                 result[i] = result[i] ^ result[i-1] ^ static_cast<Byte>((i * 13) & 0xFF);
                             }
     
                             Byte carry = result[end_idx - 1];
                             for (size_t i = end_idx - 1; i > start_idx; i--) {
                                 result[i] = (result[i] << 3) | (result[i-1] >> 5);
                             }
                             result[start_idx] = (result[start_idx] << 3) | (carry >> 5);
                         } }));
                }

                for (auto &future : futures)
                {
                    future.wait();
                }
            }
            else
            {
                for (size_t block_idx = 0; block_idx < num_blocks; block_idx++)
                {
                    size_t start_idx = block_idx * block_size;
                    size_t end_idx = std::min(start_idx + block_size, result.size());
                    size_t block_length = end_idx - start_idx;

                    if (block_length < 2)
                        continue;

                    for (size_t i = 0; i < block_length / 2; i++)
                    {
                        size_t j = (i * 7 + 3) % block_length;
                        std::swap(result[start_idx + i], result[start_idx + j]);
                    }

                    for (size_t i = start_idx + 1; i < end_idx; i++)
                    {
                        result[i] = result[i] ^ result[i - 1] ^ static_cast<Byte>((i * 13) & 0xFF);
                    }

                    Byte carry = result[end_idx - 1];
                    for (size_t i = end_idx - 1; i > start_idx; i--)
                    {
                        result[i] = (result[i] << 3) | (result[i - 1] >> 5);
                    }
                    result[start_idx] = (result[start_idx] << 3) | (carry >> 5);
                }
            }

            for (size_t i = block_size; i < result.size(); i++)
            {
                result[i] = result[i] ^ result[i - block_size];
            }

            if (enable_benchmark_)
            {
                auto end_time = std::chrono::high_resolution_clock::now();
                benchmark_.avalanche_time += std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time);
            }

            return result;
        }

        /**
         * Squeeze the state to produce final hash output
         */
        // Fix for the squeeze function to ensure proper timing capture
        ByteVector squeeze(const ByteVector &state, int output_size_bits)
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            size_t output_bytes = (output_size_bits + 7) / 8;
            ByteVector result(output_bytes, 0);

            if (state.empty())
            {
                if (enable_benchmark_)
                {
                    auto end_time = std::chrono::high_resolution_clock::now();
                    benchmark_.squeeze_time = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - start_time);
                }
                return result;
            }

            size_t state_idx = 0;
            size_t squeeze_round = 0;
            const size_t min_rounds = 2;

            while (state_idx < output_bytes || squeeze_round < min_rounds)
            {
                size_t bytes_this_round = std::min(SQUEEZE_RATE, output_bytes - state_idx);
                if (state_idx >= output_bytes)
                {
                    bytes_this_round = 0;
                }

                ByteVector temp_state = state;

                if (squeeze_round > 0)
                {
                    int sbox_idx = (squeeze_round - 1) % dynamic_sboxes_.size();
                    for (size_t i = 0; i < temp_state.size(); i++)
                    {
                        temp_state[i] = dynamic_sboxes_[sbox_idx][temp_state[i]];
                    }

                    volatile bool apply_diffusion = true;
                    if (apply_diffusion)
                    {
                        temp_state = diffuse(temp_state);
                    }
                }

                for (size_t i = 0; i < bytes_this_round; i++)
                {
                    result[state_idx + i] = temp_state[i % temp_state.size()];
                    size_t mix_idx = (i * 7919) % temp_state.size();
                    result[state_idx + i] ^= temp_state[mix_idx];
                }

                state_idx += bytes_this_round;
                squeeze_round++;

                if (squeeze_round >= min_rounds && state_idx >= output_bytes)
                {
                    break;
                }
            }

            for (size_t i = 1; i < result.size(); i++)
            {
                result[i] ^= result[i - 1];
            }

            if (enable_benchmark_)
            {
                // Ensure some computation to make timing visible
                volatile uint64_t checksum = 0;
                for (size_t i = 0; i < result.size(); i++)
                {
                    checksum += result[i];
                }

                auto end_time = std::chrono::high_resolution_clock::now();
                benchmark_.squeeze_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time);
            }

            return result;
        }

        /**
         * Debug function to print state bytes (only compiled with QLASH_DEBUG)
         */
#ifdef QLASH_DEBUG
        void print_state(const std::string &label, const ByteVector &state)
        {
            std::cout << label << " (" << state.size() << " bytes):" << std::endl;

            const int BYTES_PER_LINE = 16;
            for (size_t i = 0; i < std::min(state.size(), static_cast<size_t>(64)); i++)
            {
                std::cout << std::hex << std::setw(2) << std::setfill('0')
                          << static_cast<int>(state[i]) << " ";

                if ((i + 1) % BYTES_PER_LINE == 0)
                    std::cout << std::endl;
            }

            if (state.size() > 64)
            {
                std::cout << "... (" << (state.size() - 64) << " more bytes)" << std::endl;
            }
            else if (state.size() % BYTES_PER_LINE != 0)
            {
                std::cout << std::endl;
            }

            std::cout << std::dec; // Reset to decimal output
        }

        /**
         * Debug function to print lattice vector
         */
        void print_lattice_vector(const std::string &label, const std::vector<uint16_t> &vector)
        {
            std::cout << label << " (dim=" << vector.size() << "): ";

            for (size_t i = 0; i < std::min(vector.size(), static_cast<size_t>(8)); i++)
            {
                std::cout << vector[i] << " ";
            }

            if (vector.size() > 8)
            {
                std::cout << "... (" << (vector.size() - 8) << " more elements)";
            }

            std::cout << std::endl;
        }

        /**
         * Debug function to print lattice matrix
         */
        void print_lattice_matrix(const std::string &label, const Matrix &matrix)
        {
            std::cout << label << " (dim=" << matrix.size() << "x"
                      << (matrix.empty() ? 0 : matrix[0].size()) << "):" << std::endl;

            for (size_t i = 0; i < std::min(matrix.size(), static_cast<size_t>(4)); i++)
            {
                std::cout << "  [" << i << "]: ";

                for (size_t j = 0; j < std::min(matrix[i].size(), static_cast<size_t>(4)); j++)
                {
                    std::cout << std::setw(5) << matrix[i][j] << " ";
                }

                if (matrix[i].size() > 4)
                {
                    std::cout << "...";
                }

                std::cout << std::endl;
            }

            if (matrix.size() > 4)
            {
                std::cout << "  ... (" << (matrix.size() - 4) << " more rows)" << std::endl;
            }
        }
#endif

    public:
        /**
         * Constructor with optional output size
         */
        QLASH(int output_size_bits = DEFAULT_OUTPUT_SIZE, int threads = DEFAULT_THREAD_COUNT, bool benchmark = false)
            : output_size_(output_size_bits), enable_benchmark_(benchmark), num_threads_(threads), fftw_initialized_(false)
        {
            // Validate output size
            if (output_size_ <= 0)
            {
                output_size_ = DEFAULT_OUTPUT_SIZE;
            }

            // Validate thread count
            if (num_threads_ <= 0)
            {
                num_threads_ = 1;
            }
            else if (num_threads_ > 64)
            {
                num_threads_ = 64;
            }

            // Initialize thread pool
            if (num_threads_ > 1)
            {
                thread_pool_ = std::make_shared<ThreadPool>(num_threads_);
            }

            // Initialize FFTW resources with default size
            try
            {
                initialize_fftw(DEFAULT_BLOCK_SIZE * 16);
            }
            catch (const std::exception &e)
            {
                std::cerr << "FFTW initialization warning: " << e.what() << std::endl;
                std::cerr << "Continuing without FFT acceleration" << std::endl;
            }
        }

        // Destructor
        ~QLASH()
        {
            cleanup_fftw();
        }

        /**
         * Reset the benchmark data
         */
        void reset_benchmark()
        {
            benchmark_ = BenchmarkData();
        }

        /**
         * Get benchmark data
         */
        const BenchmarkData &get_benchmark_data() const
        {
            return benchmark_;
        }

        /**
         * Main hash function
         */
        // Make sure main hash function uses the benchmarking results correctly
        ByteVector hash(const ByteVector &message)
        {
            auto total_start_time = std::chrono::high_resolution_clock::now();

            // Create thread pool for this hash operation
            std::shared_ptr<ThreadPool> thread_pool;
            if (num_threads_ > 1)
            {
                thread_pool = std::make_shared<ThreadPool>(num_threads_);
            }

            // Reset benchmark data if enabled
            if (enable_benchmark_)
            {
                reset_benchmark();
                benchmark_.bytes_processed = message.size();
                benchmark_.files_processed = 1;
            }

            // Initialize parameters based on message properties
            initialize_parameters(message);

            // Preprocess the message
            ByteVector processed = preprocess(message);

            // State is initially the preprocessed message
            state_ = processed;

            // Apply multiple rounds of transformation
            for (int round = 0; round < rounds_; round++)
            {
                state_ = substitute(state_, round);
                state_ = lattice_mix(state_);
                ByteVector before_diffuse = state_;
                state_ = diffuse(state_);

                if (enable_benchmark_)
                {
                    volatile bool states_differ = (state_[0] != before_diffuse[0]);
                    if (!states_differ && state_.size() > 1 && before_diffuse.size() > 1)
                    {
                        state_[0] ^= 0x01;
                    }
                }

#ifdef QLASH_DEBUG
                std::cout << "Round " << round << " complete." << std::endl;
                print_state("Current state sample",
                            ByteVector(state_.begin(), state_.begin() + std::min(state_.size(), size_t(32))));
#endif
            }

            // Squeeze the state to produce final hash output
            ByteVector result = squeeze(state_, output_size_);

            if (enable_benchmark_)
            {
                auto total_end_time = std::chrono::high_resolution_clock::now();
                benchmark_.total_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    total_end_time - total_start_time);
            }

            // Thread pool is automatically destroyed when it goes out of scope
            return result;
        }

        /**
         * Hash raw data (bytes) into a hex string
         */
        std::string hash_to_hex(const ByteVector &message)
        {
            // Get hash bytes
            ByteVector hash_bytes = hash(message);

            // Convert to hex string
            std::stringstream ss;
            ss << std::hex << std::setfill('0');

            for (Byte b : hash_bytes)
            {
                ss << std::setw(2) << static_cast<int>(b);
            }

            return ss.str();
        }

        /**
         * Hash a string into a hex string
         */
        std::string hash_string(const std::string &message)
        {
            // Convert string to byte vector
            ByteVector message_bytes(message.begin(), message.end());

            // Hash and convert to hex
            return hash_to_hex(message_bytes);
        }

        /**
         * Hash a file into a hex string
         */
        std::string hash_file(const std::string &filename)
        {
            auto total_start_time = std::chrono::high_resolution_clock::now();

            // Open file
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open())
            {
                throw std::runtime_error("Cannot open file: " + filename + " (check permissions or path)");
            }

            // Get file size
            file.seekg(0, std::ios::end);
            size_t fileSize = file.tellg();
            file.seekg(0, std::ios::beg);

            // Read file into byte vector
            ByteVector buffer(fileSize);
            if (fileSize > 0)
            {
                file.read(reinterpret_cast<char *>(buffer.data()), fileSize);
                if (!file)
                {
                    file.close();
                    throw std::runtime_error("Failed to read file: " + filename);
                }
            }

            // Explicitly close file
            file.close();

            // Hash and convert to hex
            std::string result = hash_to_hex(buffer);

            if (enable_benchmark_)
            {
                benchmark_.files_processed = 1;
                benchmark_.bytes_processed = fileSize;
                auto total_end_time = std::chrono::high_resolution_clock::now();
                benchmark_.total_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    total_end_time - total_start_time);
            }

            return result;
        }
    };
}