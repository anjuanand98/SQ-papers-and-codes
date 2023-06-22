%Xnopt stores quantized regions
%Yn stores quantized points

function main
%initializing x 
num=300;%number of elements initialized in x range
Nval=[2 4 8 16 32];
eval=zeros(1,length(Nval));
edecval=zeros(1,length(Nval));
%range of x: [a,b]
a=-3;
b=3;
%x parameters
mu=0;
sigma_xsq=1;
%samples of x
delta=(b-a)/num;
xval=[a+delta/2:delta:b-delta/2];
yval=xval;
%pmf of samples of x
p=zeros(length(xval),1);
xval=[xval, b];
f1=@(xv) ((1/sqrt(2*pi*sigma_xsq))*exp(-(xv-mu).^2/(2*sigma_xsq)));
scal=1/integral(f1,a,b,'ArrayValued',true);
for i=1:length(p)
    p(i)=scal*integral(f1,xval(i)-delta/2,xval(i)+delta/2,'ArrayValued',true);
end

D1val=zeros((length(xval)-1)*length(xval)/2,2);%stores all possible intervals [alpha,beta]
k=1;%iteration over all possible (alpha,beta) pairs
for i=1:length(xval)-1%for each alpha value possible
    for j=i+1:length(xval)%for each beta value possible, given the alpha value
        D1val(k,1:2)=[xval(i) xval(j)];%(alpha,beta)
        k=k+1;
    end
end
Dnxoxval=D1val(1:length(xval)-1,2);%(xo,x) - possible x values
Dn=10^7*ones(length(Dnxoxval),Nval(end)-1);%stores Dn values for each n (2:N), for each x in (xo,x)
Xn=10^7*ones(length(Dnxoxval),Nval(end)-1);%stores X_(n-1) values for each n (1:N-1), for each x in (xo,x)

%initializing D1 for each [alpha, beta] pair in D1val
D1=zeros(size(D1val,1),1);%D1([alpha,beta])=minimum over y integral(alpha to beta (f(x,x-y)p(x)dx))
%calculating D1
for i=1:size(D1val,1)%iterate over possible (alpha,beta) pairs
    q=decoder([D1val(i,1);D1val(i,2)],xval,p,yval);
    D1(i)=encoderdistortion([D1val(i,1);D1val(i,2)],q,xval,p,yval);
end

for n=2:Nval(end)
    for i=n+1:length(xval)%Dn(xo,x(i)) - there should be sufficient levels between xo and x(i), so i starts from n, e.g., n=2 level quantization implies you need atleast three points in x 
        [minval xboundary]=nonstr_recursion_1(xval,i,n,Xn,Dn,Dnxoxval,D1val,D1);%finding Dn and X_(n-1) values for each (xo,x) for a given n level 
        %indexed such that x(i) location in Dnxoxval gives corresponding Dn and Xn values
        Dn(find((Dnxoxval==xval(i))),n-1)=minval;
        Xn(find((Dnxoxval==xval(i))),n-1)=xboundary;
    end
end
Xnoptall=zeros(Nval(end)+1,Nval(end)+1);
Ynall=zeros(Nval(end),Nval(end));
for N=Nval%number of quantization levels
[Xnopt]=nonstrategic_quantization_hloop(xval,N,p,yval,Dn,Xn,Dnxoxval)%returns quantization decision levels
Yn=decoder(Xnopt,xval,p,yval);%returns quantization representative levels
endistortion=encoderdistortion(Xnopt,Yn,xval,p,yval)%encoder distortion
dedistortion=decoderdistortion(Xnopt,Yn,xval,p,yval)%decoder distortion
%storing
Xnoptall(find(N==Nval),1:N+1)=Xnopt;
Ynall(find(N==Nval),1:N)=Yn;
eval(find(N==Nval))=endistortion;
edecval(find(N==Nval))=dedistortion;
end
%saving optimal quantization decision and representative levels, encoder
%and decoder distortions
save('gaussian_xcubed_fixed.mat','Xnoptall','Ynall','eval','edecval');
%plotting encoder and decoder distortions
f=figure;
plot(Nval,eval,'*-');
hold on;
plot(Nval,edecval,'o-');
hold off;
legend({'encoder distortion','decoder distortion'},'FontSize',14);
xlabel('N','FontSize',14);
ylabel('distortion','FontSize',14);
title('fixed rate xcubed gaussian','FontSize',14);
saveas(f,'fixedrate_xcubed_gaussian.fig');
saveas(f,'fixedrate_xcubed_gaussian.png');

function [y]=decoder(x,xval,p,yval)
%inputs: x - quantization decision levels, xval - samples of x, p - pmf of 
% xval, yval - possible representative levels
%outputs: y - quantization representative levels
N=length(x)-1;
y=zeros(1,N);
for i=1:N%iterate over each region
    disty=zeros(1,length(yval));
    in1=find(x(i)==xval);
    in2=find(x(i+1)==xval)-1;
    for j=1:length(disty)
        disty(j)=((xval(in1:in2)-yval(j)).^2)*p(in1:in2);
    end
    [md1 md2]=min(disty);
    y(i)=yval(md2);
end

function endistortion=encoderdistortion(x,y,xval,p,yval)
%inputs: x - quantization decision levels, y - quantization representative 
% levels, xval - samples of x, p - pmf of xval, yval - possible representative levels
%output: endistortion - encoder distortion
N=length(y);
k=3;
endistortion=0;
    for n=1:N%iterate over each region
        in1=find(x(n)==xval);
        in2=find(x(n+1)==xval)-1;
        endistortion=endistortion+((xval(in1:in2).^3-y(n)).^2)*p(in1:in2);  
    end
    
function dedistortion=decoderdistortion(x,y,xval,p,yval)
%inputs: x - quantization decision levels, y - quantization representative
%levels, xval - samples of x, p - pmf of xval, yval - possible representative levels
%output: dedistortion - decoder distortion
N=length(y);
dedistortion=0;
for n=1:N%iterate over each region
    in1=find(x(n)==xval);
    in2=find(x(n+1)==xval)-1;
    dedistortion=dedistortion+((xval(in1:in2)-y(n)).^2)*p(in1:in2);  
end 

function [Xnopt]=nonstrategic_quantization_hloop(xval,N,p,yval,Dn,Xn,Dnxoxval)
%inputs: xval - samples of x, N - number of quantization levels, p - pmf of xval
%output: Xnopt - quantization decision levels
%initializing x optimal end points
Xnopt=10^7*ones(N+1,1);
Xnopt(1)=xval(1);
Xnopt(end)=xval(end);
for n=N:-1:2%backward iteration to find Xnopt
    Xnopt(n)=Xn(find((Dnxoxval==Xnopt(n+1))),n-1);%X_(n-1) optimal=X_(n-1)(xo,X_n opt)
end

function [minval xboundary]=nonstr_recursion_1(xval,xind,n,Xn,Dn,Dnxoxval,D1val,D1)%returns distortion and corresponding value of x and theta
    if n==1%n=1 values are already computed in D1
        minval=D1(find(D1val(:,1)==xval(1) & D1val(:,2)==xval(xind)));
        xboundary=xval(xind);
        return;
    end
    alpharange=[n:xind-1];%indices of alpha values possible in x range (xo<alpha<x)
    arr=zeros(length(alpharange),1);%for each alpha value possible, finding minimum (D_(n)(xo,x)=min over alpha, ao<alpha<x D_(n-1)(xo,alpha)+D1(alpha,x))
    %D_n(xo<alpha<x, thetao<beta<theta)
    for arrx=1:length(alpharange)
        if n-2>=1 && Xn(find((Dnxoxval==xval(alpharange(arrx)))),n-2)~=10^7%referring to a variable for values already computed
            xboundary1=Xn(find((Dnxoxval==xval(alpharange(arrx)))),n-2);
            minval1=Dn(find((Dnxoxval==xval(alpharange(arrx)))),n-2);
        else
        [minval1 xboundary1]=nonstr_recursion_1(xval,alpharange(arrx),n-1,Xn,Dn,Dnxoxval,D1val,D1);%D_(n-1)(xo,alpha) 
        end
        %D_n(xo,x)=min over alpha, xo<alpha<x (D_(n-1)(xo,alpha)+D1(alpha,x))
        arr(arrx)=minval1+D1(find(D1val(:,1)==xval(alpharange(arrx)) & D1val(:,2)==xval(xind)));
    end
    minval=min(arr);%returning minimum distortion value
    in=find(arr==minval);
    if length(in)~=1
        in;
    end
    xboundary=xval(alpharange(in(1)));
