#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // ================== 任务1：读取测试图片 ==================
    if (argc != 2) {
        cout << "用法: ./opencv_demo 测试图片路径" << endl;
        return -1;
    }
    Mat img = imread(argv[1], IMREAD_COLOR);
    if (img.empty()) {
        cout << "❌ 无法读取图片，请检查路径！" << endl;
        return -1;
    }

    // ================== 任务2：输出图像基本信息 ==================
    cout << "=== 图像基本信息 ===" << endl;
    cout << "宽度：" << img.cols << endl;
    cout << "高度：" << img.rows << endl;
    cout << "通道数：" << img.channels() << endl;
    cout << "像素数据类型：" << typeToString(img.type()) << endl;

    // ================== 任务3：显示原图 ==================
    imshow("【原图】", img);
    waitKey(100); // 短暂延迟让窗口显示

    // ================== 任务4：转换为灰度图 ==================
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    imshow("【灰度图】", gray_img);

    // ================== 任务5：保存灰度图 ==================
    imwrite("gray_result.jpg", gray_img);
    cout << "✅ 灰度图已保存为：gray_result.jpg" << endl;

    // ================== 任务6：NumPy风格简单操作（用Mat模拟） ==================
    // 1. 输出某个像素值（以彩色图为例）
    Vec3b pixel_bgr = img.at<Vec3b>(100, 100);
    uchar pixel_gray = gray_img.at<uchar>(100, 100);
    cout << "=== 像素信息 ===" << endl;
    cout << "彩色图(100,100) B: " << (int)pixel_bgr[0] 
         << " G: " << (int)pixel_bgr[1] 
         << " R: " << (int)pixel_bgr[2] << endl;
    cout << "灰度图(100,100) 像素值: " << (int)pixel_gray << endl;

    // 2. 裁剪左上角区域并保存
    Rect roi(0, 0, 200, 200); // 左上角 200x200 区域
    Mat crop_img = img(roi);
    imwrite("crop_result.jpg", crop_img);
    cout << "✅ 裁剪区域已保存：crop_result.jpg" << endl;

    waitKey(0);  // 等待按键关闭窗口
    destroyAllWindows();
    return 0;
}