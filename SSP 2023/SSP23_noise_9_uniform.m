%Xnopt stores quantized regions
%Yn stores quantized points

function main
clc;
close all;
clear all;
kencoder=1.5;
berval=[0 0.05 0.2 0.3 0.4 0.49];
Nval=[2 4 8 16 32];
save('parameters.mat','kencoder','berval','Nval');
Xnoptall=zeros(length(berval),length(Nval),Nval(end)+1);
Ynall=zeros(length(berval),length(Nval),Nval(end));
decdistbern=zeros(length(berval),length(Nval));
encdistbern=zeros(length(berval),length(Nval));
stle=strings(1,length(berval));
for ber=berval
%initializing x 
num=300;%number of elements initialized in x range
%range of x: [a,b]
a=0;
b=1;
%samples of x
delta=(b-a)/num;
xval=[a+delta/2:delta:b-delta/2];
%reconstruction points
yval=xval;
%pmf of samples of x
p=zeros(length(xval),1);
%appending the end point so that all intervals can be taken as open on the
%right side [)
xval=[xval, b];
fun=@(xv) 1/(b-a);
scal=1/integral(fun,a,b,'ArrayValued',true); %scaling
for i=1:length(p)
    p(i)=scal*integral(fun,xval(i)-delta/2,xval(i)+delta/2,'ArrayValued',true);
end
mu=xval(1:end-1)*p;
var=((xval(1:end-1)-mu).^2)*p;
D1val=zeros((length(xval)-1)*length(xval)/2,2);%stores all possible intervals [alpha,beta]
k=1;%iteration over all possible (alpha,beta) pairs
for i=1:length(xval)-1%for each alpha value possible
    for j=i+1:length(xval)%for each beta value possible, given the alpha value
        D1val(k,1:2)=[xval(i) xval(j)];%(alpha,beta)
        k=k+1;
    end
end

for N=Nval%number of quantization levels
perr=1-(1-ber)^(log(N)/log(2));
if perr>=(N-1)/N
    error(strcat('perr condition not satisfied with ber=',num2str(ber),', N=',num2str(N)));
end
c1=1-N*perr/(N-1);
c2=perr/(N-1);
[Xnopt]=nonstrategic_quantization_hloop(xval,N,p,yval,var,mu,perr,c1,c2,D1val,kencoder)%returns quantization decision levels
Yn=decoder(Xnopt,xval,p,yval,perr,var,mu,c1,c2)%returns quantization representative levels
endistortion=encoderdistortion(Xnopt,Yn,xval,p,yval,perr,var,mu,c1,c2,kencoder)+N*(perr/(N-1))*var%encoder distortion
dedistortion=decoderdistortion(Xnopt,Yn,xval,p,yval,perr,var,mu,c1,c2)+N*(perr/(N-1))*var%decoder distortion
%storing
Xnoptall(find(ber==berval),find(N==Nval),1:N+1)=Xnopt;
Ynall(find(ber==berval),find(N==Nval),1:N)=Yn;
encdistbern(find(ber==berval),find(N==Nval))=endistortion;
decdistbern(find(ber==berval),find(N==Nval))=dedistortion;
end
stle(find(ber==berval))=strcat('ber=',num2str(ber));
end
save(strcat('uniform_str_k_',num2str(kencoder),'.mat'),'Xnoptall','Ynall','encdistbern','decdistbern','berval','kencoder','Nval');

function [y]=decoder(x,xval,p,yval,perr,var,mu,c1,c2)
%inputs: x - quantization decision levels, xval - samples of x, p - pmf of 
% xval, yval - possible representative levels
%outputs: y - quantization representative levels
N=length(x)-1;
y=zeros(1,N);
for i=1:N%iterate over each region
    disty=zeros(1,length(yval));
    in1=find(x(i)==xval);
    in2=find(x(i+1)==xval)-1;
    temp=(c1*xval(in1:in2)*p(in1:in2)+c2*mu)/(c1*sum(p(in1:in2))+c2);
    for j=1:length(disty)
        disty(j)=decoderdistortion([xval(in1) xval(in2+1)],yval(j),xval,p,yval,perr,var,mu,c1,c2);
    end
    [md1 md2]=min(disty);
    y(i)=yval(md2);
end

function endistortion=encoderdistortion(x,y,xval,p,yval,perr,var,mu,c1,c2,kencoder)
%inputs: x - quantization decision levels, y - quantization representative 
% levels, xval - samples of x, p - pmf of xval, yval - possible representative levels
%output: endistortion - encoder distortion
N=length(x);
endistortion=0;
s=((kencoder*y-mu)*(kencoder*y-mu)');
    for n=1:N-1%iterate over each region
        in1=find(x(n)==xval);
        in2=find(x(n+1)==xval)-1;
        endistortion=endistortion+((xval(in1:in2)-kencoder*y(n)).^2)*p(in1:in2);  
    end
    endistortion=c1*endistortion+c2*s;
    
function dedistortion=decoderdistortion(x,y,xval,p,yval,perr,var,mu,c1,c2)
%inputs: x - quantization decision levels, y - quantization representative
%levels, xval - samples of x, p - pmf of xval, yval - possible representative levels
%output: dedistortion - decoder distortion
N=length(x);
dedistortion=0;
s=(y-mu)*(y-mu)';
for n=1:N-1%iterate over each region
    in1=find(x(n)==xval);
    in2=find(x(n+1)==xval)-1;
    dedistortion=dedistortion+((xval(in1:in2)-y(n)).^2)*p(in1:in2);  
end 
dedistortion=c1*dedistortion+c2*s;

function [Xnopt]=nonstrategic_quantization_hloop(xval,N,p,yval,var,mu,perr,c1,c2,D1val,kencoder)
%inputs: xval - samples of x, N - number of quantization levels, p - pmf of xval
%output: Xnopt - quantization decision levels
%initializing x optimal end points
Xnopt=10^7*ones(N+1,1);
Xnopt(1)=xval(1);
Xnopt(end)=xval(end);
Dnxoxval=D1val(1:length(xval)-1,2);%(xo,x) - possible x values
Dn=10^7*ones(length(Dnxoxval),N-1);%stores Dn values for each n (2:N), for each x in (xo,x)
Xn=10^7*ones(length(Dnxoxval),N-1);%stores X_(n-1) values for each n (1:N-1), for each x in (xo,x)
%initializing D1 for each [alpha, beta] pair in D1val
D1=zeros(size(D1val,1),1);%D1([alpha,beta])=minimum over y integral(alpha to beta (f(x,x-y)p(x)dx))
%calculating D1
for i=1:size(D1val,1)%iterate over possible (alpha,beta) pairs
    q=decoder([D1val(i,1);D1val(i,2)],xval,p,yval,perr,var,mu,c1,c2);
    D1(i)=encoderdistortion([D1val(i,1);D1val(i,2)],q,xval,p,yval,perr,var,mu,c1,c2,kencoder);
end
for n=2:N
    for i=n+1:length(xval)%Dn(xo,x(i)) - there should be sufficient levels between xo and x(i), so i starts from n, e.g., n=2 level quantization implies you need atleast three points in x 
        [minval xboundary]=nonstr_recursion_1(xval,i,n,Xn,Dn,Dnxoxval,D1val,D1);%finding Dn and X_(n-1) values for each (xo,x) for a given n level 
        %indexed such that x(i) location in Dnxoxval gives corresponding Dn and Xn values
        Dn(find((Dnxoxval==xval(i))),n-1)=minval;
        Xn(find((Dnxoxval==xval(i))),n-1)=xboundary;
    end
end
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
    xboundary=xval(alpharange(in(1)));