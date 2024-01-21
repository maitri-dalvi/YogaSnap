<script>
document.addEventListener('DOMContentLoaded', function () {
    const video = document.getElementById('yoga-video');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
            video.addEventListener('loadeddata', () => {
                detectPose();
            });
        })
        .catch((error) => {
            console.error('Error accessing the camera:', error);
        });

    function detectPose() {
        const netPromise = posenet.load();

        netPromise.then(net => {
            async function poseDetectionFrame() {
                const pose = await net.estimateSinglePose(video, {
                    flipHorizontal: false
                });

                // Add your logic to handle the detected pose

                // Example: Log the keypoints to the console
                console.log(pose.keypoints);

                // Draw the pose on the canvas (optional)
                drawPose(pose);

                // Call the next frame
                requestAnimationFrame(poseDetectionFrame);
            }

            poseDetectionFrame();
        });
    }

    function drawPose(pose) {
        canvas.width = video.width;
        canvas.height = video.height;
        ctx.clearRect(0, 0, video.width, video.height);
        ctx.drawImage(video, 0, 0, video.width, video.height);

        // Customize the drawing style as needed
        ctx.fillStyle = 'red';
        ctx.strokeStyle = 'red';

        // Draw keypoints
        pose.keypoints.forEach(keypoint => {
            if (keypoint.score > 0.2) {
                ctx.beginPath();
                ctx.arc(keypoint.position.x, keypoint.position.y, 5, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            }
        });

        // Connect keypoints with lines (optional)
        // drawKeypointLines(pose, ctx);
    }

    function drawKeypointLines(pose, ctx) {
        const adjacentKeyPoints = posenet.getAdjacentKeyPoints(pose.keypoints, 0.1);

        adjacentKeyPoints.forEach((keypoints) => {
            ctx.beginPath();
            ctx.moveTo(keypoints[0].position.x, keypoints[0].position.y);
            ctx.lineTo(keypoints[1].position.x, keypoints[1].position.y);
            ctx.lineWidth = 2;
            ctx.stroke();
        });
    }
});
</script>
