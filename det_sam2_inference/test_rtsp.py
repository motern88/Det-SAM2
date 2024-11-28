import cv2


def test_rtsp_stream(rtsp_url):
    """
    测试 RTSP 视频流是否可用，显示视频帧，并获取帧率和分辨率。

    参数:
        rtsp_url (str): RTSP 流地址
    """
    print(f"正在尝试连接到 RTSP 地址: {rtsp_url}")

    # 尝试连接到 RTSP 流
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("无法打开 RTSP 流，请检查地址是否正确或网络是否正常。")
        return

    # 获取视频流的帧率和分辨率
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高度

    print(f"视频帧率: {fps} FPS")
    print(f"视频分辨率: {width} x {height}")

    print("连接成功！按 'q' 退出查看视频。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧，可能是视频流中断或结束。")
            break

        # 显示当前帧
        cv2.imshow('RTSP Stream', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("视频流测试结束。")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("视频流测试结束。")


if __name__ == "__main__":
    rtsp_url = 'rtsp://175.178.18.243:19699/'
    test_rtsp_stream(rtsp_url)