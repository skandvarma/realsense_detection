#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>

class CompressedClient : public rclcpp::Node {
private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr color_pub, depth_pub;
    std::thread receive_thread;
    
public:
    CompressedClient() : Node("compressed_client") {
        color_pub = create_publisher<sensor_msgs::msg::Image>("/camera/camera/color/image_raw", 10);
        depth_pub = create_publisher<sensor_msgs::msg::Image>("/camera/camera/aligned_depth_to_color/image_raw", 10);
        
        receive_thread = std::thread(&CompressedClient::receiveLoop, this);
    }
    
    ~CompressedClient() {
        if (receive_thread.joinable()) {
            receive_thread.join();
        }
    }
    
    void receiveLoop() {
        while (rclcpp::ok()) {
            int sock = socket(AF_INET, SOCK_STREAM, 0);
            
            struct sockaddr_in server_addr;
            server_addr.sin_family = AF_INET;
            server_addr.sin_port = htons(8889);
            inet_pton(AF_INET, "192.168.1.189", &server_addr.sin_addr);
            
            if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == 0) {
                RCLCPP_INFO(get_logger(), "Connected to server");
                
                while (rclcpp::ok()) {
                    uint32_t color_size, depth_size;
                    
                    if (recv(sock, &color_size, 4, MSG_WAITALL) != 4) break;
                    if (recv(sock, &depth_size, 4, MSG_WAITALL) != 4) break;
                    
                    std::vector<uint8_t> color_data(color_size);
                    std::vector<uint8_t> depth_data(depth_size);
                    
                    if (recv(sock, color_data.data(), color_size, MSG_WAITALL) != (ssize_t)color_size) break;
                    if (recv(sock, depth_data.data(), depth_size, MSG_WAITALL) != (ssize_t)depth_size) break;
                    
                    // Decompress images
                    cv::Mat color_img = cv::imdecode(color_data, cv::IMREAD_COLOR);
                    cv::Mat depth_img = cv::imdecode(depth_data, cv::IMREAD_UNCHANGED);
                    
                    if (!color_img.empty() && !depth_img.empty()) {
                        publishImages(color_img, depth_img);
                    }
                }
            }
            
            close(sock);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    
    void publishImages(const cv::Mat& color_img, const cv::Mat& depth_img) {
        auto timestamp = now();
        
        // Color message
        auto color_msg = std::make_shared<sensor_msgs::msg::Image>();
        color_msg->header.stamp = timestamp;
        color_msg->header.frame_id = "camera_color_optical_frame";
        color_msg->height = color_img.rows;
        color_msg->width = color_img.cols;
        color_msg->encoding = "bgr8";
        color_msg->step = color_img.cols * 3;
        color_msg->data.assign(color_img.data, color_img.data + color_img.total() * color_img.channels());
        
        // Depth message
        auto depth_msg = std::make_shared<sensor_msgs::msg::Image>();
        depth_msg->header.stamp = timestamp;
        depth_msg->header.frame_id = "camera_depth_optical_frame";
        depth_msg->height = depth_img.rows;
        depth_msg->width = depth_img.cols;
        depth_msg->encoding = "16UC1";
        depth_msg->step = depth_img.cols * 2;
        depth_msg->data.assign((uint8_t*)depth_img.data, (uint8_t*)depth_img.data + depth_img.total() * depth_img.elemSize());
        
        color_pub->publish(*color_msg);
        depth_pub->publish(*depth_msg);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto client = std::make_shared<CompressedClient>();
    rclcpp::spin(client);
    rclcpp::shutdown();
    return 0;
}
