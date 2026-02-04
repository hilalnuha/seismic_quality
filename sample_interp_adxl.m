load('timeseisadxl.mat')

%x = 0:pi/4:2*pi; 
%v = sin(x);
%xq = 0:pi/16:2*pi;

%minn=100;
%maxx=1600;
minn=100;
maxx=1600;

x = dataseis_sdxl(minn:maxx,1); 
v = dataseis_sdxl(minn:maxx,2);
seli=8;
xmin=min(dataseis_sdxl(minn:maxx,1));
xmax=max(dataseis_sdxl(minn:maxx,1));
xq = xmin:seli:xmax;

figure
%vq1 = interp1(x,v,xq);
vq1 = interp1(x,v,xq,'linear');

plot(x,v,'o',xq,vq1,':.');
xlim([xmin xmax]);
title('(Default) Linear Interpolation');