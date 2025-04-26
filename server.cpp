/**
 * QLASH API Server
 * A lightweight HTTP server providing API access to the QLASH hash function
 *
 * Standalone implementation - no external frameworks/libraries needed
 * beyond what's already included in the QLASH implementation
 */

 #include <iostream>
 #include <string>
 #include <thread>
 #include <mutex>
 #include <vector>
 #include <map>
 #include <functional>
 #include <sstream>
 #include <fstream>
 #include <cstring>
 #include <atomic>
 #include <chrono>
 #include <filesystem>
 #include <future>
 #include <queue>
 #include <condition_variable>
 #include <algorithm>
 #include <optional>
 #include <variant>
 
 // Socket headers
 #ifdef _WIN32
 #include <winsock2.h>
 #include <ws2tcpip.h>
 #pragma comment(lib, "ws2_32.lib")
 typedef SOCKET socket_t;
 #define SOCKET_ERROR_VALUE INVALID_SOCKET
 #define CLOSE_SOCKET(s) closesocket(s)
 #else
 #include <unistd.h>
 #include <sys/socket.h>
 #include <netinet/in.h>
 #include <arpa/inet.h>
 #include <signal.h>
 typedef int socket_t;
 #define SOCKET_ERROR_VALUE -1
 #define CLOSE_SOCKET(s) close(s)
 #endif
 
 // Include the QLASH library
 // Assuming it's in the same directory or in the include path
 #include "qlash.hpp"
 
 // Server configuration
 namespace config
 {
     constexpr int DEFAULT_PORT = 8080;
     constexpr int DEFAULT_THREAD_COUNT = 4;
     constexpr int CONNECTION_QUEUE_SIZE = 10;
     constexpr size_t MAX_REQUEST_SIZE = 10 * 1024 * 1024; // 10MB
     constexpr size_t MAX_UPLOAD_SIZE = 100 * 1024 * 1024; // 100MB
     constexpr size_t BUFFER_SIZE = 8192;                  // 8KB
     constexpr const char *VERSION = "1.0.0";
     constexpr const char *SERVER_NAME = "QLASH-API-Server";
 }
 
 // Atomic flag for controlling server shutdown
 std::atomic<bool> g_running{true};
 
 // Signal handler for graceful shutdown
 #ifndef _WIN32
 void signal_handler(int signal)
 {
     if (signal == SIGINT || signal == SIGTERM)
     {
         std::cout << "\nReceived shutdown signal. Stopping server..." << std::endl;
         g_running = false;
     }
 }
 #endif
 
 // HTTP utilities
 namespace http
 {
     // HTTP status codes
     enum class StatusCode
     {
         OK = 200,
         CREATED = 201,
         BAD_REQUEST = 400,
         NOT_FOUND = 404,
         METHOD_NOT_ALLOWED = 405,
         PAYLOAD_TOO_LARGE = 413,
         INTERNAL_SERVER_ERROR = 500
     };
 
     // HTTP methods
     enum class Method
     {
         GET,
         POST,
         PUT,
         OPTIONS,
         UNKNOWN
     };
 
     // Convert HTTP method string to enum
     Method parse_method(const std::string &method_str)
     {
         if (method_str == "GET")
             return Method::GET;
         if (method_str == "POST")
             return Method::POST;
         if (method_str == "PUT")
             return Method::PUT;
         if (method_str == "OPTIONS")
             return Method::OPTIONS;
         return Method::UNKNOWN;
     }
 
     // Convert StatusCode to string
     std::string status_code_to_string(StatusCode code)
     {
         switch (code)
         {
         case StatusCode::OK:
             return "200 OK";
         case StatusCode::CREATED:
             return "201 Created";
         case StatusCode::BAD_REQUEST:
             return "400 Bad Request";
         case StatusCode::NOT_FOUND:
             return "404 Not Found";
         case StatusCode::METHOD_NOT_ALLOWED:
             return "405 Method Not Allowed";
         case StatusCode::PAYLOAD_TOO_LARGE:
             return "413 Payload Too Large";
         case StatusCode::INTERNAL_SERVER_ERROR:
             return "500 Internal Server Error";
         default:
             return "500 Internal Server Error";
         }
     }
 
     // HTTP request
     struct Request
     {
         Method method;
         std::string path;
         std::string version;
         std::map<std::string, std::string> headers;
         std::string body;
         std::map<std::string, std::string> query_params;
 
         // Parse query parameters from the path
         void parse_query_params()
         {
             size_t query_start = path.find('?');
             if (query_start != std::string::npos)
             {
                 std::string query_string = path.substr(query_start + 1);
                 path = path.substr(0, query_start);
 
                 size_t pos = 0;
                 while (pos < query_string.length())
                 {
                     size_t amp_pos = query_string.find('&', pos);
                     if (amp_pos == std::string::npos)
                         amp_pos = query_string.length();
 
                     std::string param = query_string.substr(pos, amp_pos - pos);
                     size_t eq_pos = param.find('=');
                     if (eq_pos != std::string::npos)
                     {
                         std::string key = param.substr(0, eq_pos);
                         std::string value = param.substr(eq_pos + 1);
                         // URL decode value if needed
                         query_params[key] = value;
                     }
 
                     pos = amp_pos + 1;
                 }
             }
         }
     };
 
     // HTTP response
     struct Response
     {
         StatusCode status;
         std::map<std::string, std::string> headers;
         std::string body;
 
         Response() : status(StatusCode::OK)
         {
             // Default headers
             headers["Server"] = std::string(config::SERVER_NAME) + "/" + config::VERSION;
             headers["Content-Type"] = "text/plain";
             headers["Access-Control-Allow-Origin"] = "*";
             headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS";
             headers["Access-Control-Allow-Headers"] = "Content-Type";
         }
 
         Response(StatusCode code) : Response()
         {
             status = code;
         }
 
         // Set JSON content type
         void set_json()
         {
             headers["Content-Type"] = "application/json";
         }
 
         // Convert response to string
         std::string to_string() const
         {
             std::stringstream ss;
 
             // Status line
             ss << "HTTP/1.1 " << status_code_to_string(status) << "\r\n";
 
             // Headers
             for (const auto &[key, value] : headers)
             {
                 ss << key << ": " << value << "\r\n";
             }
 
             // Add Content-Length header
             ss << "Content-Length: " << body.length() << "\r\n";
 
             // End of headers
             ss << "\r\n";
 
             // Body
             ss << body;
 
             return ss.str();
         }
     };
 
     // Parse HTTP request from raw string
     std::optional<Request> parse_request(const std::string &request_str)
     {
         Request req;
 
         // Extract request line, headers, and body
         std::istringstream request_stream(request_str);
         std::string request_line;
         if (!std::getline(request_stream, request_line))
         {
             return std::nullopt;
         }
 
         // Parse request line
         std::istringstream request_line_stream(request_line);
         std::string method_str;
         if (!(request_line_stream >> method_str >> req.path >> req.version))
         {
             return std::nullopt;
         }
         req.method = parse_method(method_str);
 
         // Parse headers
         std::string header_line;
         while (std::getline(request_stream, header_line) && header_line != "\r")
         {
             auto delimiter_pos = header_line.find(':');
             if (delimiter_pos != std::string::npos)
             {
                 std::string key = header_line.substr(0, delimiter_pos);
                 std::string value = header_line.substr(delimiter_pos + 1);
 
                 // Trim whitespace
                 value.erase(0, value.find_first_not_of(" \t\r\n"));
                 value.erase(value.find_last_not_of(" \t\r\n") + 1);
 
                 req.headers[key] = value;
             }
         }
 
         // Extract the remaining body (handle special case for POST with Content-Length)
         if (req.method == Method::POST)
         {
             auto content_length_it = req.headers.find("Content-Length");
             if (content_length_it != req.headers.end())
             {
                 size_t content_length = std::stoul(content_length_it->second);
                 // Read the remaining body
                 if (content_length > 0 && content_length <= config::MAX_REQUEST_SIZE)
                 {
                     // Skip the \r\n after headers if any
                     auto pos = request_str.find("\r\n\r\n");
                     if (pos != std::string::npos)
                     {
                         size_t body_start = pos + 4;
                         if (body_start < request_str.length())
                         {
                             req.body = request_str.substr(body_start);
                         }
                     }
                 }
             }
         }
 
         // Parse query parameters
         req.parse_query_params();
 
         return req;
     }
 
     // URL encoding/decoding functions
     std::string url_encode(const std::string &input)
     {
         std::ostringstream escaped;
         escaped.fill('0');
         escaped << std::hex;
 
         for (char c : input)
         {
             if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
             {
                 escaped << c;
             }
             else
             {
                 escaped << '%' << std::setw(2) << int((unsigned char)c);
             }
         }
 
         return escaped.str();
     }
 
     std::string url_decode(const std::string &input)
     {
         std::string result;
         for (size_t i = 0; i < input.length(); ++i)
         {
             if (input[i] == '%')
             {
                 if (i + 2 < input.length())
                 {
                     int value;
                     std::istringstream is(input.substr(i + 1, 2));
                     if (is >> std::hex >> value)
                     {
                         result += static_cast<char>(value);
                         i += 2;
                     }
                     else
                     {
                         result += input[i];
                     }
                 }
                 else
                 {
                     result += input[i];
                 }
             }
             else if (input[i] == '+')
             {
                 result += ' ';
             }
             else
             {
                 result += input[i];
             }
         }
         return result;
     }
 }
 
 // API Router
 class Router
 {
 public:
     using HandlerFunc = std::function<http::Response(const http::Request &)>;
 
     // Register a handler for a specific path and method
     void add_route(const std::string &path, http::Method method, HandlerFunc handler)
     {
         routes_[path][method] = handler;
     }
 
     // Dispatch a request to the appropriate handler
     http::Response dispatch(const http::Request &request)
     {
         // Check if route exists
         auto route_it = routes_.find(request.path);
         if (route_it != routes_.end())
         {
             // Check if method handler exists
             auto method_it = route_it->second.find(request.method);
             if (method_it != route_it->second.end())
             {
                 // Execute handler
                 return method_it->second(request);
             }
             else if (request.method == http::Method::OPTIONS)
             {
                 // Handle OPTIONS method for CORS preflight
                 http::Response response;
                 response.status = http::StatusCode::OK;
                 response.body = "";
                 return response;
             }
             else
             {
                 // Method not allowed
                 return http::Response(http::StatusCode::METHOD_NOT_ALLOWED);
             }
         }
 
         // Route not found
         return http::Response(http::StatusCode::NOT_FOUND);
     }
 
 private:
     std::map<std::string, std::map<http::Method, HandlerFunc>> routes_;
 };
 
 // QLASH API handlers
 namespace api
 {
     // Get server info
     http::Response get_info(const http::Request &request)
     {
         http::Response response;
         response.set_json();
 
         std::stringstream json;
         json << "{\n";
         json << "  \"server\": \"" << config::SERVER_NAME << "\",\n";
         json << "  \"version\": \"" << config::VERSION << "\",\n";
         json << "  \"qlash_version\": \"" << QLASH_VERSION << "\",\n";
         json << "  \"qlash_build\": \"" << QLASH_BUILD << "\",\n";
         json << "  \"endpoints\": [\n";
         json << "    { \"path\": \"/api/info\", \"method\": \"GET\", \"description\": \"Get server information\" },\n";
         json << "    { \"path\": \"/api/hash\", \"method\": \"POST\", \"description\": \"Hash text data\" },\n";
         json << "    { \"path\": \"/api/hash/hex\", \"method\": \"POST\", \"description\": \"Hash text data and return hex representation\" },\n";
         json << "    { \"path\": \"/api/benchmark\", \"method\": \"POST\", \"description\": \"Run benchmark on input data\" }\n";
         json << "  ]\n";
         json << "}";
 
         response.body = json.str();
         return response;
     }
 
     // Hash text data
     http::Response hash_text(const http::Request &request)
     {
         http::Response response;
 
         // Check if request body is too large
         if (request.body.size() > config::MAX_REQUEST_SIZE)
         {
             response.status = http::StatusCode::PAYLOAD_TOO_LARGE;
             response.body = "Request body too large";
             return response;
         }
 
         try
         {
             // Extract parameters
             int output_size = 512; // Default
             int threads = config::DEFAULT_THREAD_COUNT;
             bool benchmark = false;
 
             // Extract query parameters
             auto output_size_it = request.query_params.find("output_size");
             if (output_size_it != request.query_params.end())
             {
                 output_size = std::stoi(output_size_it->second);
                 if (output_size <= 0 || output_size > 4096)
                 {
                     output_size = 512; // Reset to default if invalid
                 }
             }
 
             auto threads_it = request.query_params.find("threads");
             if (threads_it != request.query_params.end())
             {
                 threads = std::stoi(threads_it->second);
                 if (threads <= 0 || threads > 64)
                 {
                     threads = config::DEFAULT_THREAD_COUNT; // Reset to default if invalid
                 }
             }
 
             auto benchmark_it = request.query_params.find("benchmark");
             if (benchmark_it != request.query_params.end() && benchmark_it->second == "true")
             {
                 benchmark = true;
             }
 
             // Create QLASH instance
             qlash::QLASH hasher(output_size, threads, benchmark);
 
             // Convert string to byte vector
             qlash::ByteVector data(request.body.begin(), request.body.end());
 
             // Compute hash
             qlash::ByteVector hash_result = hasher.hash(data);
 
             // Set response as binary data
             response.headers["Content-Type"] = "application/octet-stream";
             response.body = std::string(hash_result.begin(), hash_result.end());
 
             // Add benchmark data if requested
             if (benchmark)
             {
                 const auto &bench_data = hasher.get_benchmark_data();
                 std::stringstream bench_header;
                 bench_header << bench_data.total_time.count() << ","
                              << bench_data.preprocessing_time.count() << ","
                              << bench_data.lattice_time.count() << ","
                              << bench_data.avalanche_time.count() << ","
                              << bench_data.sbox_time.count() << ","
                              << bench_data.squeeze_time.count() << ","
                              << bench_data.bytes_processed << ","
                              << bench_data.get_throughput_mbps();
                 response.headers["X-Benchmark-Data"] = bench_header.str();
             }
 
             return response;
         }
         catch (const std::exception &e)
         {
             response.status = http::StatusCode::INTERNAL_SERVER_ERROR;
             response.body = std::string("Error: ") + e.what();
             return response;
         }
     }
 
     // Hash text data and return hex
     http::Response hash_text_hex(const http::Request &request)
     {
         http::Response response;
 
         // Check if request body is too large
         if (request.body.size() > config::MAX_REQUEST_SIZE)
         {
             response.status = http::StatusCode::PAYLOAD_TOO_LARGE;
             response.body = "Request body too large";
             return response;
         }
 
         try
         {
             // Extract parameters
             int output_size = 512; // Default
             int threads = config::DEFAULT_THREAD_COUNT;
             bool benchmark = false;
 
             // Extract query parameters
             auto output_size_it = request.query_params.find("output_size");
             if (output_size_it != request.query_params.end())
             {
                 output_size = std::stoi(output_size_it->second);
                 if (output_size <= 0 || output_size > 4096)
                 {
                     output_size = 512; // Reset to default if invalid
                 }
             }
 
             auto threads_it = request.query_params.find("threads");
             if (threads_it != request.query_params.end())
             {
                 threads = std::stoi(threads_it->second);
                 if (threads <= 0 || threads > 64)
                 {
                     threads = config::DEFAULT_THREAD_COUNT; // Reset to default if invalid
                 }
             }
 
             auto benchmark_it = request.query_params.find("benchmark");
             if (benchmark_it != request.query_params.end() && benchmark_it->second == "true")
             {
                 benchmark = true;
             }
 
             // Create QLASH instance
             qlash::QLASH hasher(output_size, threads, benchmark);
 
             // Hash the input text
             std::string hash_result = hasher.hash_string(request.body);
 
             // Set response
             response.headers["Content-Type"] = "text/plain";
             response.body = hash_result;
 
             // Add benchmark data if requested
             if (benchmark)
             {
                 const auto &bench_data = hasher.get_benchmark_data();
                 std::stringstream bench_header;
                 bench_header << bench_data.total_time.count() << ","
                              << bench_data.preprocessing_time.count() << ","
                              << bench_data.lattice_time.count() << ","
                              << bench_data.avalanche_time.count() << ","
                              << bench_data.sbox_time.count() << ","
                              << bench_data.squeeze_time.count() << ","
                              << bench_data.bytes_processed << ","
                              << bench_data.get_throughput_mbps();
                 response.headers["X-Benchmark-Data"] = bench_header.str();
             }
 
             return response;
         }
         catch (const std::exception &e)
         {
             response.status = http::StatusCode::INTERNAL_SERVER_ERROR;
             response.body = std::string("Error: ") + e.what();
             return response;
         }
     }
 
     // Run benchmark
     http::Response run_benchmark(const http::Request &request)
     {
         http::Response response;
         response.set_json();
 
         try
         {
             // Extract parameters
             int output_size = 512; // Default
             int threads = config::DEFAULT_THREAD_COUNT;
             int iterations = 5;             // Default
             size_t data_size = 1024 * 1024; // Default 1MB
 
             // Extract query parameters
             auto output_size_it = request.query_params.find("output_size");
             if (output_size_it != request.query_params.end())
             {
                 output_size = std::stoi(output_size_it->second);
                 if (output_size <= 0 || output_size > 4096)
                 {
                     output_size = 512; // Reset to default if invalid
                 }
             }
 
             auto threads_it = request.query_params.find("threads");
             if (threads_it != request.query_params.end())
             {
                 threads = std::stoi(threads_it->second);
                 if (threads <= 0 || threads > 64)
                 {
                     threads = config::DEFAULT_THREAD_COUNT; // Reset to default if invalid
                 }
             }
 
             auto iterations_it = request.query_params.find("iterations");
             if (iterations_it != request.query_params.end())
             {
                 iterations = std::stoi(iterations_it->second);
                 if (iterations <= 0 || iterations > 100)
                 {
                     iterations = 5; // Reset to default if invalid
                 }
             }
 
             auto data_size_it = request.query_params.find("data_size");
             if (data_size_it != request.query_params.end())
             {
                 data_size = std::stoul(data_size_it->second);
                 if (data_size == 0 || data_size > 100 * 1024 * 1024)
                 {
                     data_size = 1024 * 1024; // Reset to default if invalid
                 }
             }
 
             // Create QLASH instance
             qlash::QLASH hasher(output_size, threads, true);
 
             // Generate random test data
             std::vector<qlash::Byte> test_data(data_size);
             std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
             std::uniform_int_distribution<int> dist(0, 255);
             for (size_t i = 0; i < data_size; ++i)
             {
                 test_data[i] = static_cast<qlash::Byte>(dist(rng));
             }
 
             // Run benchmark
             std::vector<qlash::BenchmarkData> results;
             for (int i = 0; i < iterations; ++i)
             {
                 hasher.reset_benchmark();
                 hasher.hash(test_data);
                 results.push_back(hasher.get_benchmark_data());
             }
 
             // Calculate average results
             qlash::BenchmarkData avg;
             avg.preprocessing_time = std::chrono::microseconds(0);
             avg.lattice_time = std::chrono::microseconds(0);
             avg.avalanche_time = std::chrono::microseconds(0);
             avg.sbox_time = std::chrono::microseconds(0);
             avg.squeeze_time = std::chrono::microseconds(0);
             avg.total_time = std::chrono::microseconds(0);
             avg.bytes_processed = 0;
 
             for (const auto &result : results)
             {
                 avg.preprocessing_time += result.preprocessing_time;
                 avg.lattice_time += result.lattice_time;
                 avg.avalanche_time += result.avalanche_time;
                 avg.sbox_time += result.sbox_time;
                 avg.squeeze_time += result.squeeze_time;
                 avg.total_time += result.total_time;
                 avg.bytes_processed += result.bytes_processed;
             }
 
             // Divide by number of iterations
             avg.preprocessing_time = std::chrono::microseconds(avg.preprocessing_time.count() / iterations);
             avg.lattice_time = std::chrono::microseconds(avg.lattice_time.count() / iterations);
             avg.avalanche_time = std::chrono::microseconds(avg.avalanche_time.count() / iterations);
             avg.sbox_time = std::chrono::microseconds(avg.sbox_time.count() / iterations);
             avg.squeeze_time = std::chrono::microseconds(avg.squeeze_time.count() / iterations);
             avg.total_time = std::chrono::microseconds(avg.total_time.count() / iterations);
             avg.bytes_processed = avg.bytes_processed / iterations;
 
             // Build JSON response
             std::stringstream json;
             json << "{\n";
             json << "  \"config\": {\n";
             json << "    \"output_size\": " << output_size << ",\n";
             json << "    \"threads\": " << threads << ",\n";
             json << "    \"data_size\": " << data_size << ",\n";
             json << "    \"iterations\": " << iterations << "\n";
             json << "  },\n";
             json << "  \"results\": {\n";
             json << "    \"total_time_ms\": " << avg.total_time.count() / 1000.0 << ",\n";
             json << "    \"preprocessing_time_ms\": " << avg.preprocessing_time.count() / 1000.0 << ",\n";
             json << "    \"lattice_time_ms\": " << avg.lattice_time.count() / 1000.0 << ",\n";
             json << "    \"avalanche_time_ms\": " << avg.avalanche_time.count() / 1000.0 << ",\n";
             json << "    \"sbox_time_ms\": " << avg.sbox_time.count() / 1000.0 << ",\n";
             json << "    \"squeeze_time_ms\": " << avg.squeeze_time.count() / 1000.0 << ",\n";
             json << "    \"throughput_mbps\": " << avg.get_throughput_mbps() << ",\n";
             json << "    \"breakdown\": {\n";
             json << "      \"preprocessing\": " << (100.0 * avg.preprocessing_time.count() / avg.total_time.count()) << "%,\n";
             json << "      \"lattice\": " << (100.0 * avg.lattice_time.count() / avg.total_time.count()) << "%,\n";
             json << "      \"avalanche\": " << (100.0 * avg.avalanche_time.count() / avg.total_time.count()) << "%,\n";
             json << "      \"sbox\": " << (100.0 * avg.sbox_time.count() / avg.total_time.count()) << "%,\n";
             json << "      \"squeeze\": " << (100.0 * avg.squeeze_time.count() / avg.total_time.count()) << "%\n";
             json << "    }\n";
             json << "  }\n";
             json << "}";
 
             response.body = json.str();
             return response;
         }
         catch (const std::exception &e)
         {
             response.status = http::StatusCode::INTERNAL_SERVER_ERROR;
             response.body = std::string("{\"error\": \"") + e.what() + "\"}";
             return response;
         }
     }
 }
 
 // HTTP Server
 class HttpServer
 {
 public:
     HttpServer(int port = config::DEFAULT_PORT, int worker_threads = config::DEFAULT_THREAD_COUNT)
         : port_(port),
           worker_threads_(worker_threads),
           server_socket_(SOCKET_ERROR_VALUE)
     {
 
         // Initialize router with API endpoints
         router_.add_route("/api/info", http::Method::GET, api::get_info);
         router_.add_route("/api/hash", http::Method::POST, api::hash_text);
         router_.add_route("/api/hash/hex", http::Method::POST, api::hash_text_hex);
         router_.add_route("/api/benchmark", http::Method::POST, api::run_benchmark);
 
 // Initialize socket library on Windows
 #ifdef _WIN32
         WSADATA wsaData;
         if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
         {
             throw std::runtime_error("Failed to initialize Winsock");
         }
 #endif
     }
 
     ~HttpServer()
     {
         stop();
 
 // Cleanup socket library on Windows
 #ifdef _WIN32
         WSACleanup();
 #endif
     }
 
     void start()
     {
         // Create socket
         server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
         if (server_socket_ == SOCKET_ERROR_VALUE)
         {
             throw std::runtime_error("Failed to create socket");
         }
 
         // Set socket options
         int opt = 1;
         if (setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt)) < 0)
         {
             CLOSE_SOCKET(server_socket_);
             throw std::runtime_error("Failed to set socket options");
         }
 
         // Bind socket
         struct sockaddr_in address;
         std::memset(&address, 0, sizeof(address));
         address.sin_family = AF_INET;
         address.sin_addr.s_addr = INADDR_ANY;
         address.sin_port = htons(port_);
 
         if (bind(server_socket_, (struct sockaddr *)&address, sizeof(address)) < 0)
         {
             CLOSE_SOCKET(server_socket_);
             throw std::runtime_error("Failed to bind socket");
         }
 
         // Listen
         if (listen(server_socket_, config::CONNECTION_QUEUE_SIZE) < 0)
         {
             CLOSE_SOCKET(server_socket_);
             throw std::runtime_error("Failed to listen on socket");
         }
 
         // Create worker threads
         for (int i = 0; i < worker_threads_; ++i)
         {
             worker_threads_vec_.emplace_back(&HttpServer::worker_thread, this);
         }
 
         std::cout << "QLASH API Server started on port " << port_
                   << " with " << worker_threads_ << " worker threads" << std::endl;
         std::cout << "Press Ctrl+C to stop the server" << std::endl;
 
         // Main accept loop
         int addr_len = sizeof(address);
         while (g_running)
         {
             // Accept connection
             socket_t client_socket = accept(server_socket_, (struct sockaddr *)&address, (socklen_t *)&addr_len);
             if (client_socket == SOCKET_ERROR_VALUE)
             {
                 if (!g_running)
                     break; // Server is shutting down
                 std::cerr << "Error accepting connection" << std::endl;
                 continue;
             }
 
             // Queue the client socket for processing by a worker thread
             {
                 std::lock_guard<std::mutex> lock(queue_mutex_);
                 client_queue_.push(client_socket);
                 queue_cv_.notify_one();
             }
         }
 
         // Wait for worker threads to finish
         for (auto &thread : worker_threads_vec_)
         {
             if (thread.joinable())
             {
                 thread.join();
             }
         }
     }
 
     void stop()
     {
         g_running = false;
 
         // Close server socket to interrupt accept()
         if (server_socket_ != SOCKET_ERROR_VALUE)
         {
             CLOSE_SOCKET(server_socket_);
             server_socket_ = SOCKET_ERROR_VALUE;
         }
 
         // Notify all worker threads to exit
         {
             std::lock_guard<std::mutex> lock(queue_mutex_);
             queue_cv_.notify_all();
         }
 
         // Wait for worker threads to finish
         for (auto &thread : worker_threads_vec_)
         {
             if (thread.joinable())
             {
                 thread.join();
             }
         }
 
         worker_threads_vec_.clear();
 
         // Close any remaining client sockets
         {
             std::lock_guard<std::mutex> lock(queue_mutex_);
             while (!client_queue_.empty())
             {
                 socket_t client_socket = client_queue_.front();
                 client_queue_.pop();
                 CLOSE_SOCKET(client_socket);
             }
         }
     }
 
 private:
     // Worker thread function
     void worker_thread()
     {
         while (g_running)
         {
             socket_t client_socket;
 
             // Get client socket from queue
             {
                 std::unique_lock<std::mutex> lock(queue_mutex_);
                 queue_cv_.wait(lock, [this]
                                { return !client_queue_.empty() || !g_running; });
 
                 if (!g_running && client_queue_.empty())
                 {
                     break;
                 }
 
                 if (client_queue_.empty())
                 {
                     continue;
                 }
 
                 client_socket = client_queue_.front();
                 client_queue_.pop();
             }
 
             // Handle client
             handle_client(client_socket);
 
             // Close socket
             CLOSE_SOCKET(client_socket);
         }
     }
 
     // Handle client connection
     void handle_client(socket_t client_socket)
     {
         // Read request
         std::string request_str;
         char buffer[config::BUFFER_SIZE];
         int bytes_read;
         size_t total_bytes_read = 0;
         bool headers_complete = false;
         size_t content_length = 0;
         size_t header_end = std::string::npos;
 
         while ((bytes_read = recv(client_socket, buffer, sizeof(buffer) - 1, 0)) > 0)
         {
             buffer[bytes_read] = '\0';
             request_str.append(buffer, bytes_read);
             total_bytes_read += bytes_read;
 
             // Check for end of headers if not already found
             if (!headers_complete)
             {
                 header_end = request_str.find("\r\n\r\n");
                 if (header_end != std::string::npos)
                 {
                     headers_complete = true;
 
                     // Parse Content-Length header
                     std::string headers = request_str.substr(0, header_end);
                     size_t content_length_pos = headers.find("Content-Length: ");
                     if (content_length_pos != std::string::npos)
                     {
                         size_t value_start = content_length_pos + 16; // "Content-Length: " length
                         size_t value_end = headers.find("\r\n", value_start);
                         if (value_end != std::string::npos)
                         {
                             std::string length_str = headers.substr(value_start, value_end - value_start);
                             content_length = std::stoul(length_str);
 
                             // Check if request is too large
                             if (content_length > config::MAX_REQUEST_SIZE)
                             {
                                 // Send 413 Payload Too Large
                                 http::Response response(http::StatusCode::PAYLOAD_TOO_LARGE);
                                 response.body = "Request body too large";
                                 std::string response_str = response.to_string();
                                 send(client_socket, response_str.c_str(), response_str.length(), 0);
                                 return;
                             }
                         }
                     }
 
                     // If we have all the data, break
                     if (content_length == 0 || total_bytes_read >= header_end + 4 + content_length)
                     {
                         break;
                     }
                 }
             }
             else if (total_bytes_read >= header_end + 4 + content_length)
             {
                 // If we have all the data (headers + body), break
                 break;
             }
 
             // Check if request is too large
             if (total_bytes_read > config::MAX_REQUEST_SIZE)
             {
                 // Send 413 Payload Too Large
                 http::Response response(http::StatusCode::PAYLOAD_TOO_LARGE);
                 response.body = "Request too large";
                 std::string response_str = response.to_string();
                 send(client_socket, response_str.c_str(), response_str.length(), 0);
                 return;
             }
         }
 
         // Parse request
         auto request_opt = http::parse_request(request_str);
         if (!request_opt)
         {
             // Bad request
             http::Response response(http::StatusCode::BAD_REQUEST);
             response.body = "Invalid request format";
             std::string response_str = response.to_string();
             send(client_socket, response_str.c_str(), response_str.length(), 0);
             return;
         }
 
         // Handle request
         http::Response response = router_.dispatch(request_opt.value());
 
         // Send response
         std::string response_str = response.to_string();
         send(client_socket, response_str.c_str(), response_str.length(), 0);
     }
 
     int port_;
     int worker_threads_;
     socket_t server_socket_;
     std::vector<std::thread> worker_threads_vec_;
     std::queue<socket_t> client_queue_;
     std::mutex queue_mutex_;
     std::condition_variable queue_cv_;
     Router router_;
 };
 
 // Main entry point
 int main(int argc, char *argv[])
 {
     // Parse command line arguments
     int port = config::DEFAULT_PORT;
     int threads = config::DEFAULT_THREAD_COUNT;
 
     for (int i = 1; i < argc; ++i)
     {
         std::string arg = argv[i];
         if (arg == "--port" || arg == "-p")
         {
             if (i + 1 < argc)
             {
                 port = std::stoi(argv[++i]);
             }
         }
         else if (arg == "--threads" || arg == "-t")
         {
             if (i + 1 < argc)
             {
                 threads = std::stoi(argv[++i]);
             }
         }
         else if (arg == "--help" || arg == "-h")
         {
             std::cout << "QLASH API Server v" << config::VERSION << std::endl;
             std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
             std::cout << "Options:" << std::endl;
             std::cout << "  --port, -p <port>      Port to listen on (default: " << config::DEFAULT_PORT << ")" << std::endl;
             std::cout << "  --threads, -t <num>    Number of worker threads (default: " << config::DEFAULT_THREAD_COUNT << ")" << std::endl;
             std::cout << "  --help, -h             Show this help message" << std::endl;
             return 0;
         }
     }
 
 // Set up signal handler on Unix-like systems
 #ifndef _WIN32
     signal(SIGINT, signal_handler);
     signal(SIGTERM, signal_handler);
 #endif
 
     try
     {
         // Create and start server
         HttpServer server(port, threads);
         server.start();
     }
     catch (const std::exception &e)
     {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     }
 
     return 0;
 }