clear
close
clc
load('timeseisadxl.mat')

%x = 0:pi/4:2*pi; 
%v = sin(x);
%xq = 0:pi/16:2*pi;

%minn=100;
%maxx=1600;
minn=1;
maxx=1896;
xo = dataseis_sdxl(minn:maxx,1); 
vo = dataseis_sdxl(minn:maxx,2);

ss=4;
x = dataseis_sdxl(minn:ss:maxx,1); 
v = dataseis_sdxl(minn:ss:maxx,2);
seli=1;
xmin=min(x);
xmax=max(x);
xq = xmin:seli:xmax;
i=1;
for ii=1:max(length(x))
    xy=find(xq==xo(ii));
    if (numel(xy)>0)
        x0(i)=ii;
        xx(i)=xy(1);
        i=i+1;
    end
end
%%
figure
%vq1 = interp1(x,v,xq);
vq1 = interp1(x,v,xq,'linear');

plot(xo,vo,'-o',xq,vq1,':.');
legend('Actual','Interpolation')
xlim([2084 2584]);
%title('(Default) Linear Interpolation');
vy=vq1(xx)';
v0=vo(x0);
grid
rmsee=sqrt(mse(vy,v0))
r2=rsquare(vy,v0)*100
xlabel("Time Index")
ylabel("Magnitude")