clc;
close all;
clear all;

load('gaussian_parameterscubed.mat');
load(strcat('gaussian_str_xcubed.mat'));

berval(4:5)=[];
encdistbern(4:5,:)=[];
decdistbern(4:5,:)=[];
fdfig=figure;
stle=strings(1,length(berval));

ber=berval(1);
plot(log(Nval)./log(2),decdistbern(find(ber==berval),:),'-o');
stle(find(ber==berval))=strcat('p_{b}=',num2str(ber));
hold on;

ber=berval(2);
plot(log(Nval)./log(2),decdistbern(find(ber==berval),:),'-*');
stle(find(ber==berval))=strcat('p_{b}=',num2str(ber));
hold on;

ber=berval(3);
plot(log(Nval)./log(2),decdistbern(find(ber==berval),:),'-s');
stle(find(ber==berval))=strcat('p_{b}=',num2str(ber));
hold on;

ber=berval(4);
plot(log(Nval)./log(2),decdistbern(find(ber==berval),:),'-^');
stle(find(ber==berval))=strcat('p_{b}=',num2str(ber));
% hold on;
% 
% ber=berval(5);
% plot(log(Nval)./log(2),decdistbern(find(ber==berval),:),'-*');
% stle(find(ber==berval))=strcat('ber=',num2str(ber));
% hold on;
% 
% ber=berval(6);
% plot(log(Nval)./log(2),decdistbern(find(ber==berval),:),'-*');
% stle(find(ber==berval))=strcat('ber=',num2str(ber));

hold off;
grid on;
lgd=legend(stle);
lgd.FontSize=14;
lgd.NumColumns=2;
xlabel('rate (in bits)','FontSize',14)
ylabel('decoder distortion','FontSize',14)
saveas(fdfig,'decdist_gaussian_xcubed.png')


%encoder
fefig=figure;

ber=berval(1);
plot(log(Nval)./log(2),encdistbern(find(ber==berval),:),'-o');
hold on;

ber=berval(2);
plot(log(Nval)./log(2),encdistbern(find(ber==berval),:),'-*');
hold on;

ber=berval(3);
plot(log(Nval)./log(2),encdistbern(find(ber==berval),:),'-s');
hold on;

ber=berval(4);
plot(log(Nval)./log(2),encdistbern(find(ber==berval),:),'-^');
% hold on;
% 
% ber=berval(5);
% plot(log(Nval)./log(2),encdistbern(find(ber==berval),:),'-*');
% stle(find(ber==berval))=strcat('ber=',num2str(ber));
% hold on;
% 
% ber=berval(6);
% plot(log(Nval)./log(2),encdistbern(find(ber==berval),:),'-*');
% stle(find(ber==berval))=strcat('ber=',num2str(ber));

hold off;
grid on;
lgd=legend(stle);
lgd.FontSize=14;
lgd.NumColumns=2;
xlabel('rate (in bits)','FontSize',14)
ylabel('encoder distortion','FontSize',14)
saveas(fefig,'encdist_gaussian_xcubed.png')


for N=Nval
    fq=figure;
    qle=strings(1,N);
    xt=zeros(1,N+1);
    for i=1:length(berval)
        plot(berval(i)*ones(1,N+1),reshape(Xnoptall(i,find(N==Nval),1:N+1),1,N+1),"o");
        hold on;
        qle(i)=strcat('ber=',num2str(berval(i)));
    end
    hold off;
    xlabel('p_{err}','FontSize',14);
    ylabel('quantizer decision level','FontSize',14)
    saveas(fq,strcat('quantizers_xcubed_N_',num2str(N),'.png'))
end
