function showFacialLandmarks( Y, shift_x, opt )
% draw nose, mouth, jaw landmarks part-by-part
plot(Y(1:2:14)+shift_x, Y(2:2:14), opt);
hold on;
plot(Y(15:2:24)+shift_x, Y(16:2:24), opt);
plot(Y(25:2:52)+shift_x, Y(26:2:52), opt);
plot(Y([25,51])+shift_x, Y([26,52]), opt);
plot(Y(53:2:end)+shift_x, Y(54:2:end), opt);
plot(Y([53,75])+shift_x, Y([54,76]), opt);
axis([-1, 2.5, -1.2 0.2]);
hold off;

end

