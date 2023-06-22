function main
clc;
close all;
clear all;

%% parameters
Mval=[2];


encdistM=zeros(length(Mval),1);
decdistM=zeros(length(Mval),1);
% a=0;
% b=1;
% f1=@(xv) 1/(b-a);



a=-5;
b=5;
mux=0;
sigma_xsq=1;
f1=@(xv) ((1/sqrt(2*pi*sigma_xsq))*exp(-(xv-mux).^2/(2*sigma_xsq)));




% xsamp=linspace(a+eps,b-eps,15);
xsamp=[-4 linspace(-3,3,30) 4];
for M=Mval
 
xminit=nchoosek(xsamp,M-1);
xminit=[ones(size(xminit,1),1)*a xminit ones(size(xminit,1),1)*b];
rn=20;
xminit=xminit(randi(size(xminit,1),1,20),:)
% rn=size(xminit,1); % number of initializations USE A GRID

xrandinit=zeros(M+1,rn); % all initializations
xrm=zeros(M+1,rn); % final quantizer values for all initializations
erm=zeros(1,rn); % encoder distortions for all initializations
yrm=zeros(M,rn); % final quantizer representative values for all initializations
drm=zeros(1,rn); % decoder distortions for all initializations
% dervrn=zeros(rn,10000,M-1);
exitflag=zeros(1,rn);
derend=zeros(M-1,rn);
tic
for r=1:rn
flag=1;
xmiter=zeros(M+1,100); % quantizer values for each iteration given an initial point
endist=zeros(1,100); % encoder distortions for each iteration given an initial point
frendist=zeros(1,100); % fractional difference in encoder distortions for each iteration given an initial point
dedist=zeros(1,100); % decoder distortions for each iteration given an initial point
derv=zeros(10000,M-1);
iter=1;
xrandinit(:,r)=xminit(r,:)';
xmiter(:,1)=xminit(r,:)';
xm=xmiter(:,1)';
ym=reconstruction(xm,f1);
dist_enc=encoderdistortion(xm,ym,f1);
dist_dec=decoderdistortion(xm,ym,f1);
endist(1)=dist_enc;
dedist(1)=dist_dec;
delta=10;
tic
while flag
    for i=2:M
        ym=reconstruction(xm,f1);
        der=derivative(xm,ym,f1,i);
        derv(iter,i-1)=der;
        temp=xm(i)-delta*der;
        xm1=xm;
        xm1(i)=temp;
        ym=reconstruction(xm1,f1);
        d1=encoderdistortion(xm1,ym,f1);

        if (temp>xm(i-1) && temp<xm(i+1) ) && d1<dist_enc
            xm(i)=temp;
        else
            [xm]=check(xm,f1,delta,der,dist_enc,i);
        end
        ym=reconstruction(xm,f1);
        dist_enc=encoderdistortion(xm,ym,f1);
    end
    xmtemp=xm;

    ymtemp=reconstruction(xmtemp,f1);
    dist_enctemp=encoderdistortion(xmtemp,ymtemp,f1);

    if iter>1
    if (endist(iter) == endist(iter-1))
        flag=0;
        exitflag(r)=2;
    end
    end
    if all(abs(derv(iter,:)) <10^-7 ) 
        flag=0;
        exitflag(r)=1;
    else

    iter=iter+1;
    xm=xmtemp;
    ym=ymtemp;
    xmiter(:,iter)=xm;
    dist_enc=dist_enctemp;
    endist(iter)=dist_enc;
    dedist(iter)=decoderdistortion(xm,ym,f1);
    end
end
toc
derend(:,r)=derv(iter,:);
xrm(:,r)=xm;
erm(r)=dist_enc;
yrm(:,r)=reconstruction(xm,f1);
drm(r)=decoderdistortion(xm,yrm(:,r),f1);
% dervrn(r,1:iter,:)=derv(1:iter,:);
disp(strcat('M = ',num2str(M),', r = ',num2str(r)))
exitf=exitflag(r);
exitf
xm
ym
dist_enc
end

toc;

[in1 in2]=min(erm);
xm=xrm(:,in2)
ym=reconstruction(xm,f1)
dist_enc=encoderdistortion(xm,ym,f1)
dist_dec=decoderdistortion(xm,ym,f1)

save(strcat('M',num2str(M),'noiseless_xcubed_gaussian.mat'),'xm','ym','dist_enc','dist_dec','erm','xrm','yrm','drm','derend','xrandinit')


end


function [xm]=check(xm,f1,delta,der,dist_enc,i)
while delta>10^-7
    delta=delta/10;
    temp=xm(i)-delta*der;
    xm1=xm;
    xm1(i)=temp;
    ym=reconstruction(xm1,f1);
    d1=encoderdistortion(xm1,ym,f1);
    if (temp>xm1(i-1) && temp<xm1(i+1) ) && d1<dist_enc
        xm(i)=temp;
        break;
    end
end

function [dist_dec]=decoderdistortion(xm,ym,f1)
M=length(xm)-1;

dist_dec=0;
for i=1:M 
    f5=@(xv) (xv-ym(i))^2*f1(xv);
    dist_dec=dist_dec+integral(f5,xm(i),xm(i+1),'ArrayValued',true);
end

function [dist_enc]=encoderdistortion(xm,ym,f1)
M=length(xm)-1;

dist_enc=0;
for i=1:M 
    f5=@(xv) (xv^3-ym(i))^2*f1(xv);
    dist_enc=dist_enc+integral(f5,xm(i),xm(i+1),'ArrayValued',true);  
end


function [ym]=reconstruction(xm,f1)
M=length(xm)-1;
f2=@(xv) xv*f1(xv);
ym=zeros(1,M);
for i=1:M
    num=integral(f2,xm(i),xm(i+1),'ArrayValued',true);
    den=integral(f1,xm(i),xm(i+1),'ArrayValued',true);
    ym(i)=num/den;
end


function [der]=derivative(xm,ym,f1,i)
M=length(xm)-1;
der=0;
    der=(xm(i)^3-ym(i-1))^2*f1(xm(i));
    der=der-(xm(i)^3-ym(i))^2*f1(xm(i));
    f3_1=@(xv) (xv^3-ym(i-1))*f1(xv);
    f3_2=@(xv) (xv^3-ym(i))*f1(xv);

    if xm(i-1)~=xm(i)
        dyixi=f1(xm(i))*(xm(i)-ym(i-1))/(integral(f1,xm(i-1),xm(i)));
        der=der-2*dyixi*integral(f3_1,xm(i-1),xm(i),'ArrayValued',true);
    end
    if xm(i)~=xm(i+1)
        dyi1xi=-f1(xm(i))*(xm(i)-ym(i))/(integral(f1,xm(i),xm(i+1)));
        der=der-2*dyi1xi*integral(f3_2,xm(i),xm(i+1),'ArrayValued',true);
    end
    