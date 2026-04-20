
import cv2
import numpy as np


class AprilTagPoseEstimator:
    def __init__(self, tag_size):
        self.tag_size = float(tag_size)

    def estimate_pose(self, tag, camera_matrix, dist_coeffs):
        """
        根据 AprilTag 的四个角点，用 solvePnP 估计 tag 相对相机的位姿。
        """
        half = self.tag_size / 2.0

        object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )

        image_points = np.array(tag.corners, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        return success, rvec, tvec


def draw_apriltags(image, tags):
    output = image.copy()

    for tag in tags:
        corners = np.array(tag.corners, dtype=np.int32)
        center = tuple(np.array(tag.center, dtype=np.int32))

        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            cv2.line(output, p1, p2, (0, 255, 0), 2)

        for i, corner in enumerate(corners):
            point = tuple(corner)
            cv2.circle(output, point, 4, (0, 0, 255), -1)
            cv2.putText(
                output,
                str(i),
                (point[0] + 5, point[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        cv2.circle(output, center, 5, (255, 0, 0), -1)
        cv2.putText(
            output,
            f"id={tag.tag_id}",
            (center[0] + 10, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    return output


def draw_pose_axes(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length=0.05):
    axis_points_3d = np.array(
        [
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, -axis_length],
        ],
        dtype=np.float32,
    )

    image_points, _ = cv2.projectPoints(
        axis_points_3d,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    image_points = image_points.reshape(-1, 2).astype(np.int32)

    origin_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    origin_2d, _ = cv2.projectPoints(
        origin_3d,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    origin = tuple(origin_2d.reshape(-1, 2).astype(np.int32)[0])

    cv2.line(image, origin, tuple(image_points[0]), (0, 0, 255), 2)
    cv2.line(image, origin, tuple(image_points[1]), (0, 255, 0), 2)
    cv2.line(image, origin, tuple(image_points[2]), (255, 0, 0), 2)

    return image
