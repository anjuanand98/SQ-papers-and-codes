function main
clc;
close all;
clear all;

%% parameters
Mval=[2];

% discretizing theta, theta in [at,bt], mean mut, variance sigma_thsq
at=-5;
bt=5;
% thval=linspace(at+(bt-at)/(2*nt),bt-(bt-at)/(2*nt),nt); 
mut=0;
sigma_thsq=1;
thval1=linspace(at,mut-2*sigma_thsq,1);
thval2=linspace(mut-2*sigma_thsq,mut-sigma_thsq,2);
thval3=linspace(mut-sigma_thsq,mut+sigma_thsq,3);
thval4=linspace(mut+sigma_thsq,mut+2*sigma_thsq,2);
thval5=linspace(mut+2*sigma_thsq,bt,1);
thval=[thval1(2:end) thval2(2:end) thval3(2:end) thval4(2:end) thval5(2:end-1)];
thval=[thval1 thval2(2:end) thval3(2:end) thval4(2:end) thval5(2:end-1)];
thval=[-1;1];
nt=length(thval);
% pdf of theta
pth=zeros(1,length(thval));
f12=@(tv) ((1/sqrt(2*pi*sigma_thsq))*exp(-(tv-mut).^2/(2*sigma_thsq)));
sct=integral(f12,at,bt,'ArrayValued',true);
pth(1)=integral(f12,at,thval(1)+(thval(2)-thval(1))/2,'ArrayValued',true)/sct;
for i=2:length(thval)-1
    pth(i)=integral(f12,thval(i)-(thval(i)-thval(i-1))/2,thval(i)+(thval(i+1)-thval(i))/2,'ArrayValued',true)/sct;
end
pth(length(thval))=integral(f12,thval(end)-(thval(end)-thval(end-1))/2,bt,'ArrayValued',true)/sct;


encdistM=zeros(length(Mval),1);
decdistM=zeros(length(Mval),1);
% a=0;
% b=1;
% f1=@(xv) 1/(b-a);



a=-5;
b=5;
rho=0;
mux=0; % mean of source X
sigma_xsq=1; % variance of source X

mux_corr=mux+rho*(sigma_xsq/sigma_thsq)^(1/2)*(thval(:)-mut); % mean of X conditional on theta 
sigma_xsq_corr=(1-rho^2)*sigma_xsq; % variance of X conditional on theta 
f1=@(xv,i) ((1/sqrt(2*pi*sigma_xsq_corr))*exp(-(xv-mux_corr(i)).^2/(2*sigma_xsq_corr)))*pth(i); % pdf of X conditional on theta


[xva,yva]=ndgrid(linspace(a+0.01,b-0.01,30));
edist=zeros(size(xva));
der1=zeros(size(xva));
der2=zeros(size(xva));
xsamp=linspace(a,b,12);
for i=1:size(xva,1)
    for j=1:size(yva,1)
        x0=[a xva(i,j) b; a yva(i,j) b];
        [y0]=reconstruction(x0,f1,thval);
        edist(i,j)=encoderdistortion(x0,y0,f1,thval);
        der1(i,j)=derivative(x0,y0,f1,2,1,thval);
        der2(i,j)=derivative(x0,y0,f1,2,2,thval);
    end
end
surf(xva,yva,edist)
xlabel('x')
ylabel('y')
colorbar
figure
surf(xva,yva,der1)
colorbar
figure
surf(xva,yva,der2)
colorbar


xsamp=linspace(a,b,12);

for M=Mval

    x0init=nchoosek(xsamp(2:end-1),M-1);
x0init=[a*ones(size(x0init,1),1) x0init b*ones(size(x0init,1),1)];
rn=20;
xrn1=randi(size(x0init,1),rn,length(thval));
xminit=zeros(rn,length(thval),M+1);
for r=1:rn

xminit(r,:,:)=x0init(xrn1(r,:)',:);
end


rn=size(xminit,1); % number of initializations USE A GRID

xrandinit=zeros(length(thval),M+1,rn); % all initializations
xrm=zeros(length(thval),M+1,rn); % final quantizer values for all initializations
erm=zeros(1,rn); % encoder distortions for all initializations
yrm=zeros(M,rn); % final quantizer representative values for all initializations
drm=zeros(1,rn); % decoder distortions for all initializations
% dervrn=zeros(length(thval),M-1,10000,rn);
exitflag=zeros(1,rn);
derend=zeros(length(thval),M-1,rn);
tic
for r=1:rn
flag=1;
xmiter=zeros(length(thval),M+1,100); % quantizer values for each iteration given an initial point
endist=zeros(1,100); % encoder distortions for each iteration given an initial point
frendist=zeros(1,100); % fractional difference in encoder distortions for each iteration given an initial point
dedist=zeros(1,100); % decoder distortions for each iteration given an initial point
derv=zeros(length(thval),M-1,100);
iter=1;
xrandinit(:,:,r)=xminit(r,:,:);
xmiter(:,:,1)=reshape(xminit(r,:,:),length(thval),M+1);
xm=xmiter(:,:,1);
ym=reconstruction(xm,f1,thval);
dist_enc=encoderdistortion(xm,ym,f1,thval);
dist_dec=decoderdistortion(xm,ym,f1,thval);
endist(1)=dist_enc;
dedist(1)=dist_dec;
delta=1;
tic;
while flag
    for t=1:length(thval)
    for i=2:M
        der=derivative(xm,ym,f1,i,t,thval);
        derv(t,i-1,iter)=der;
        temp=xm(t,i)-delta*der;
        xm1=xm;
        xm1(t,i)=temp;
        ym=reconstruction(xm1,f1,thval);
        d1=encoderdistortion(xm1,ym,f1,thval);

        if (temp>xm(t,i-1) && temp<xm(t,i+1)) && d1<dist_enc
            xm(t,i)=temp;
            
        else
            [xm]=check(xm,f1,delta,der,dist_enc,i,t,thval);
        end
        ym=reconstruction(xm,f1,thval);
        dist_enc=encoderdistortion(xm,ym,f1,thval);
    end
    end
    xmtemp=xm;
%     xmtemp=xmcheck(xmtemp,xmtemp,delta,der);% ensuring the constraints are satisfied
    ymtemp=reconstruction(xmtemp,f1,thval);
    dist_enctemp=encoderdistortion(xmtemp,ymtemp,f1,thval);
%     frendist(iter)=(dist_enc-dist_enctemp)/dist_enc;
        if iter>1
        if (endist(iter) == endist(iter-1))
            flag=0;
            exitflag(r)=2;
        end
        end
    if all(abs(derv(:,:,iter)) <10^-7 ) 
        flag=0;
        exitflag(r)=1;
    else

    iter=iter+1;
    xm=xmtemp;
    ym=ymtemp;
    xmiter(:,:,iter)=xm;
    dist_enc=dist_enctemp;
    endist(iter)=dist_enc;
    dedist(iter)=decoderdistortion(xm,ym,f1,thval);
    end
end
toc
derend(:,:,r)=derv(:,:,iter);
xrm(:,:,r)=xm;
erm(r)=dist_enc;
yrm(:,r)=reconstruction(xm,f1,thval);
drm(r)=decoderdistortion(xm,yrm(:,r),f1,thval);
% dervrn(r,1:iter,:)=derv(1:iter,:);
disp(strcat('M = ',num2str(M),', r = ',num2str(r),', rho = ',num2str(rho)))
exitf=exitflag(r);
exitf
xm
ym
dist_enc

end
toc
[in1 in2]=min(erm);
xm=xrm(:,:,in2)
ym=reconstruction(xm,f1,thval)
dist_enc=encoderdistortion(xm,ym,f1,thval)
dist_dec=decoderdistortion(xm,ym,f1,thval)

save(strcat('xthetaM',num2str(M),'rho',num2str(rho),'noiseless_xcubed_gaussian.mat'),'xm','ym','dist_enc','dist_dec','erm','xrm','yrm','drm','derend','xrandinit')
% derend=zeros(M-1,rn);
% for r=1:rn
%     temp=dervrn(1:M-1,1:length(find(dervrn(:,:,r)~=0))/(M-1),r);
%     derend(:,r)=temp(:,end);
% end
end

function [xm]=check(xm,f1,delta,der,dist_enc,i,t,thval)
while delta>10^-7
    delta=delta/10;
    temp=xm(t,i)-delta*der;
    xm1=xm;
    xm1(t,i)=temp;
    ym=reconstruction(xm1,f1,thval);
    d1=encoderdistortion(xm1,ym,f1,thval);
    if (temp>xm1(t,i-1) && temp<xm1(t,i+1) ) && d1<dist_enc
        xm(t,i)=temp;
        break;
    end
end

function [dist_dec]=decoderdistortion(xthetam,ym,f1,thval)
M=size(xthetam,2)-1;
dist_dec=0;
for i=1:M
    for k=1:length(thval)
        f1temp= @(xv) f1(xv,k);
        f5=@(xv) (xv-ym(i))^2*f1temp(xv);
        dist_dec=dist_dec+integral(f5,xthetam(k,i),xthetam(k,i+1),'ArrayValued',true);
    end
end

function [dist_enc]=encoderdistortion(xthetam,ym,f1,thval)
M=size(xthetam,2)-1;
dist_enc=0;
for i=1:M
    for k=1:length(thval)
        f1temp= @(xv) f1(xv,k);
        f5=@(xv) (xv+thval(k)-ym(i))^2*f1temp(xv);
        dist_enc=dist_enc+integral(f5,xthetam(k,i),xthetam(k,i+1),'ArrayValued',true);
    end
end


function [ym]=reconstruction(xthetam,f1,thval)
M=size(xthetam,2)-1;

ym=zeros(1,M);
for i=1:M
    num=0;
    den=0;
    for j=1:length(thval)
        f1temp= @(xv) f1(xv,j);
        f2=@(xv) xv*f1temp(xv);
        num=num+integral(f2,xthetam(j,i),xthetam(j,i+1),'ArrayValued',true);
        den=den+integral(f1temp,xthetam(j,i),xthetam(j,i+1),'ArrayValued',true);
    end
    if den~=0
        ym(i)=num/den;
    else
        ym(i)=(1/size(xthetam,1))*sum(xthetam(:,i));
    end
end

function [der]=derivative(xm,ym,f1,i,t,thval)
M=size(xm,2)-1;

der=0;
den1=0;
den2=0;
num=f1(xm(t,i),t);
for th=1:size(thval)
    den1=den1+integral(@(xv) f1(xv,th),xm(th,i-1),xm(th,i));
    den2=den2+integral(@(xv) f1(xv,th),xm(th,i),xm(th,i+1));
end
den1=den1;
den2=den2;



    der=(xm(t,i)+thval(t)-ym(i-1))^2*f1(xm(t,i),t);
    der=der-(xm(t,i)+thval(t)-ym(i))^2*f1(xm(t,i),t);

dyixi=(xm(t,i)-ym(i-1))*num/den1;
dyi1xi=-(xm(t,i)-ym(i))*num/den2;
for th=1:length(thval)
    f3_1=@(xv) (xv+thval(th)-ym(i-1))*f1(xv,th);
    f3_2=@(xv) (xv+thval(th)-ym(i))*f1(xv,th);
    if xm(th,i-1)~=xm(th,i)        
        der=der-2*dyixi*integral(f3_1,xm(th,i-1),xm(th,i),'ArrayValued',true);
    end
    if xm(th,i)~=xm(th,i+1)        
        der=der-2*dyi1xi*integral(f3_2,xm(th,i),xm(th,i+1),'ArrayValued',true);
    end
end   
   
    
function [f22] = f22fn(x,thval,f1,a,b)

M=length(x)/length(thval)+1;
x=[a*ones(length(thval),1) reshape(x,M-1,length(thval))' b*ones(length(thval),1)];
[ym]=reconstruction(x,thval,f1,mux);

x=x';
x=x(:);

f22=0;
for i=1:M
    for t=1:length(thval)
            f22=f22+integral(@(xv)(xv+thval(t)-ym(i))^2*f1(xv,t),x((t-1)*(M+1)+i),x((t-1)*(M+1)+i+1),'ArrayValued',true);
    end
end


