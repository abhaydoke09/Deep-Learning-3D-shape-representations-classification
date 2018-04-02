function CreateAvi(Yp_test, Y_test, filename)
%% create AVI video file for visualizing ground-truth and predicted result
h=figure;
movegui(h, 'onscreen');
title('Landmark Prediction')
vidObj = VideoWriter(filename);
vidObj.Quality = 100;
vidObj.FrameRate = 100;
open(vidObj);

for i = 1:size(Yp_test, 1)
    showFacialLandmarks( Y_test(i,:), 0, 'r-' ); % draw ground-truth (red)
    hold on;
    showFacialLandmarks( Yp_test(i,:), 1.5, 'b-' ); % draw predicted result (blue)
    hold off;
    axis([-1, 2.5, -1.2 0.2]);

    drawnow;
    currFrame = getframe;
    writeVideo(vidObj,currFrame);
end
close(vidObj);
end
