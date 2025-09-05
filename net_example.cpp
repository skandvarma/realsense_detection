#include <librealsense2/rs.hpp>
#include <librealsense2-net/rs_net.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    try
    {
        std::string ip_address = "192.168.1.189";
        std::string output_dir = "/tmp/realsense_frames";

        if (argc > 1) {
            ip_address = argv[1];
        }

        // Create output directory
        mkdir(output_dir.c_str(), 0755);

        // Create network device
        rs2::net_device dev(ip_address);

        // Create a context and add device
        rs2::context ctx;
        dev.add_to(ctx);

        // Create pipeline with this context
        rs2::pipeline pipe(ctx);

        // Configure and start pipeline
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

        auto profile = pipe.start(cfg);

        // Setup alignment
        rs2::align align(RS2_STREAM_COLOR);

        // Extract and save intrinsics
        auto color_stream = profile.get_stream(RS2_STREAM_COLOR);
        auto color_intrinsics = color_stream.as<rs2::video_stream_profile>().get_intrinsics();

        std::ofstream intrinsics_file(output_dir + "/intrinsics.txt");
        intrinsics_file << color_intrinsics.fx << " " << color_intrinsics.fy << " "
                       << color_intrinsics.ppx << " " << color_intrinsics.ppy << " "
                       << color_intrinsics.width << " " << color_intrinsics.height << std::endl;
        intrinsics_file.close();

        std::cout << "Streaming from RealSense at " << ip_address << std::endl;
        std::cout << "Frames saved to: " << output_dir << std::endl;
        std::cout << "Press ESC to exit" << std::endl;

        // Create OpenCV windows
        cv::namedWindow("Color Stream", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Depth Stream", cv::WINDOW_AUTOSIZE);

        int frame_count = 0;

        while (true)
        {
            rs2::frameset frames = pipe.wait_for_frames();

            // Align depth to color
            frames = align.process(frames);

            rs2::video_frame color = frames.get_color_frame();
            rs2::depth_frame depth = frames.get_depth_frame();

            if (color && depth) {
                // Convert RealSense frames to OpenCV Mat
                cv::Mat color_mat(cv::Size(color.get_width(), color.get_height()),
                                 CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
                cv::Mat depth_mat(cv::Size(depth.get_width(), depth.get_height()),
                                 CV_16UC1, (void*)depth.get_data(), cv::Mat::AUTO_STEP);

                // Convert depth for visualization
                cv::Mat depth_display;
                depth_mat.convertTo(depth_display, CV_8UC1, 0.03);

                // Display frames
                cv::imshow("Color Stream", color_mat);
                cv::imshow("Depth Stream", depth_display);

                // Save frames for Python consumption (every 10th frame to avoid too much I/O)
                if (frame_count % 10 == 0) {
                    cv::imwrite(output_dir + "/latest_color.jpg", color_mat);
                    cv::imwrite(output_dir + "/latest_depth.png", depth_mat);

                    // Write timestamp
                    std::ofstream timestamp_file(output_dir + "/timestamp.txt");
                    timestamp_file << std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count() << std::endl;
                    timestamp_file.close();
                }

                frame_count++;

                if (frame_count % 30 == 0) {
                    std::cout << "Frame " << frame_count << " captured" << std::endl;
                }
            }

            // Break on ESC key
            if (cv::waitKey(1) == 27) break;
        }

        pipe.stop();
        cv::destroyAllWindows();
    }
    catch (const rs2::error &e)
    {
        std::cerr << "RealSense error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;

private:
    void extract_intrinsics()
    {
        try {
            auto color_stream = profile_.get_stream(RS2_STREAM_COLOR);
            auto intrinsics = color_stream.as<rs2::video_stream_profile>().get_intrinsics();

            std::cout << "Camera intrinsics:" << std::endl;
            std::cout << "  fx: " << intrinsics.fx << ", fy: " << intrinsics.fy << std::endl;
            std::cout << "  ppx: " << intrinsics.ppx << ", ppy: " << intrinsics.ppy << std::endl;
            std::cout << "  Resolution: " << intrinsics.width << "x" << intrinsics.height << std::endl;

        } catch (const std::exception &e) {
            std::cerr << "Could not extract intrinsics: " << e.what() << std::endl;
        }
    }

    std::string ip_address_;
    std::string output_dir_;
    std::shared_ptr<rs2::context> ctx_;
    std::shared_ptr<rs2::net_device> net_dev_;
    std::shared_ptr<rs2::pipeline> pipe_;
    rs2::pipeline_profile profile_;
    std::shared_ptr<rs2::align> align_;
};

int main(int argc, char *argv[])
{
    std::string ip_address = "192.168.1.189";

    if (argc > 1) {
        ip_address = argv[1];
    }

    RealSenseStreamer streamer(ip_address);

    if (streamer.initialize()) {
        streamer.run();
    }

    return 0;
}