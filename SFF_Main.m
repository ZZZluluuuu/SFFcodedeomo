%% Shape From Focus
clc;clear;close all;
addpath(genpath('Functions'))
%% 3D-Reconstruction
tic
%Path of image sequences
Name='007';
ImagePath='F:\BGA chip M Plan10X\';
mainPath=[ImagePath,Name,'\'];
%Path of results
path='H:\测试';
mkdir([path '\' '1_DP'])
mkdir([path '\' '2_All_in_focus'])
q=9;%% Param of FMO
[xSteps,ySteps,zSteps,stepDistance,pixelThan,x_pin,y_pin] = Re_SetParam(mainPath);
%Save params
suofang=0.2;
suofang1=1;
%Load the network
load 'noisenet_10cov.mat';

focus=1:zSteps;focus=double(focus);
%% Main code
for x_i=0:xSteps-1
    for y_i=0:ySteps-1
        str_y=num2str(y_i,'%03d');
        str_x=num2str(x_i,'%03d');
        subPath=[mainPath,str_x,str_y];
        files=dir([subPath,'\*.jpg']);
        I=im2double(imread([subPath,'\',files(1).name]));
        [m,n,o]=size(I);
        m1=floor(m/q);
        n1=floor(n/q);

        C1=cell(zSteps,1); %Save the grayscale map
        FV_result=zeros(m1,n1);
        yuantu_linshi=zeros(m,n,o);

        %% 1.Calculate the focus value
        Ini_FV=zeros(m1,n1,zSteps);
        for k=1:zSteps
            try
                img=im2double(imread([subPath '\' num2str(k,'%03d') '.jpg']));
                C1{k}=img;
                if o>1
                    img=im2gray(img);
                end
                Ini_FV(:,:,k)=fun_GLV_Local(img,q);
            catch
                C1{k}=yuantu_linshi;
                Ini_FV(:,:,k)=FV_result;
            end
        end

        %% 2.Fit the focus curve
        [~, Ini_DP] = gauss3P_halfmax(focus, Ini_FV);

        %% 3.Noise classification
        A_FV=scale_volume(Ini_FV);
        Flag=fun_Classify(A_FV,net);

        %% 4.Denoise the FV
        Ini_DP(Flag==0)=nan;
        Ini_DP1=inpaint_nans(Ini_DP,4);
        Ini_DP1=imresize(Ini_DP1,[m,n],'bilinear');
        Ini_DP1(Ini_DP1<1)=1;

        %% 5.Draw all-in focua images
        if o==3
            AIF=ones(m,n,3);
            for i1=1:m
                for j1=1:n
                    AIF(i1,j1,:)=C1{round(Ini_DP1(i1,j1))}(i1,j1,:);
                end
            end
        else
            AIF=ones(m,n);
            for i1=1:m
                for j1=1:n
                    AIF(i1,j1)=C1{round(Ini_DP1(i1,j1))}(i1,j1);
                end
            end
        end
        clear C1 i1 j1 m n Ini_DP

        %% 6.Save the file

        save([path,'\','1_DP\', str_x,str_y,'.mat'],'Ini_DP1');
        savePath=[path,'\','2_AIF\',str_x,str_y,'.jpg'];
        imwrite(AIF,savePath)
        [str_x,str_y]

    end
end
time1=toc

%% Save the results
tic
fun_MergeDeep([path,'\1_DP\'],xSteps,ySteps,y_pin,x_pin,suofang1,path)
fun_MergeImg([path,'\2_AIF\'],xSteps,ySteps,y_pin,x_pin,suofang1,path)

load ([path,'\','MergeDeep.mat'])

MergeDeep1 = imresize(MergeDeep,suofang);
Zmax=floor(1.5*max(max(MergeDeep1))*stepDistance);

[m,n]=size(MergeDeep1);
xt1=(0:pixelThan:(n-1)*pixelThan)/(suofang*suofang1);
yt1=(0:pixelThan:(m-1)*pixelThan)/(suofang*suofang1);
[xt1,yt1] = meshgrid(xt1,yt1);

angle1=160;
angle2=45;
beilv=1;
%%% Initial depth map
figure(1)
surf(xt1,yt1,MergeDeep1*stepDistance)
view([angle1,angle2]) 
axis equal;
zlim([0,Zmax]);
set(gca,'DataAspectRatio',[1 1 beilv])
set(gca,'FontWeight','bold','FontName','Times New Roman','FontSize',14,'LineWidth',1);
set(gcf,'color','w');
shading interp;
ax = gca;
ax.XDir = 'reverse';
xlabel('X/(μm)'),ylabel('Y/(μm)'),zlabel('Z/(μm)');
box on
grid off
time2=toc

%% Final depth map
MergeDeep2=Fun_fitdepthsurface(MergeDeep1,0.015);%%Set the params
save([path,'\', 'MergeDeep2.mat'],'MergeDeep2');
figure(2)
surf(xt1,yt1,MergeDeep2*stepDistance)
view([angle1,angle2]) 
axis equal;
zlim([0,Zmax]);
set(gca,'DataAspectRatio',[1 1 beilv])
set(gca,'FontWeight','bold','FontName','Times New Roman','FontSize',14,'LineWidth',1);
set(gcf,'color','w');
shading interp;
ax = gca;
ax.XDir = 'reverse';
xlabel('X/(μm)'),ylabel('Y/(μm)'),zlabel('Z/(μm)');
box on
grid off

%% SSM
SSM=fun_SSM(MergeDeep2);
%% All-in focus image
AIF=im2double(imread([path,'\MergeImage.jpg']));
AIF = imresize(AIF,suofang);
figure(3);
warp(xt1,yt1,MergeDeep2*stepDistance,AIF)
set(gca,'XLim',[0 (n-1)*pixelThan/(suofang*suofang1)]);set(gca,'YLim',[0 (m-1)*pixelThan/(suofang*suofang1)]);set(gca,'ZLim',[0 zSteps*stepDistance]);
view([angle1,angle2])
axis equal;
zlim([0,Zmax]);
set(gca,'DataAspectRatio',[1 1 beilv])
set(gca,'FontWeight','bold','FontName','Times New Roman','FontSize',14,'LineWidth',1);
set(gcf,'color','w');
shading interp;
ax = gca;
ax.XDir = 'reverse';
xlabel('X/(μm)'),ylabel('Y/(μm)'),zlabel('Z/(μm)');
box on
grid off
