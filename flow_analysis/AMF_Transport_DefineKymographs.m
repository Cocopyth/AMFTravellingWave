%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AMF_Transport_DefineKymographs %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% AMF
%   Transport
%       DefineKymographs: from sequence of images
%                           to .mat (individual files - savedir)
%                           to .mat (global file - logdir - CurrentListings.mat)
%                           to .png (expdir - DefineKymographs)
%       MakeKymographs: from sequence of images
%                           to .mat (.mat - savedir)
%                           to .tif (image - expdir - Kymographs)
%                           to .png (Figure - expdir - KymoFig)
% Version PMMH PhB - 240128 (Myristate) - last update: 240319

%% INTIALIZE

% Cleaning
clear all
close all

% Base folders
sourcedir='/home/philippe/Documents/AMF_Transport/Data/ToAnalyze';
logdir='/home/philippe/Documents/AMF_Transport/Data/AnalysisLogFiles';
expdir='/home/philippe/Documents/AMF_Transport/Exp/Plates';
savedir='/home/philippe/Documents/AMF_Transport/Data/Plates';
saveindir='/home/philippe/Documents/AMF_Transport/Data/Indiv';
movdir='/home/philippe/Documents/AMF_Transport/Movies';


%% DEFINE KYMOGRAPHS

% Load Current Listings
cd(logdir);
load('CurrentListings.mat');

% Pick parameters
L=20; %um
px2um=3.45/50; % Magnification x50
Lpx=L/px2um;
% ChoicePlate=find([listseq.plate]==408);

for i=1:size(listmov,2)
    
    % Initialize
    close all

    % Disp platename
    listmov(i).name
    
    % Disp First Image
    cd(listmov(i).dirname);
    cd('Img');
    A=imread(listmov(i).listim(1).name);
    h=figure;
    imagesc(A);
    colormap(gray);
    hold on

    % DefineKymographs 
    % <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    % Pick and display 2 points (if never proceed)
    [xtmp,ytmp]=ginput(2);

    % Load mat file (if already stored)
    %cd(strcat(savedir,'/',listmov(i).platename,'/DefineKymographs'));
    %load(strcat(listmov(i).name(3:end),'.mat'));

    % >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    plot(xtmp(1),ytmp(1),'vr');
    plot(xtmp(2),ytmp(2),'^r','markerfacecolor','r');


    % Adjust Length of the kymograph
    l12=((diff(xtmp)).^2+(diff(ytmp)).^2).^(1/2);
    xre=[xtmp(1) xtmp(1)+Lpx/l12*diff(xtmp)];
    yre=[ytmp(1) ytmp(1)+Lpx/l12*diff(ytmp)];

    plot(round(xre),round(yre),'ob');
    xr=[xre(1)+[0:1:round(Lpx)].*(diff(xre))/Lpx];
    yr=[yre(1)+[0:1:round(Lpx)].*(diff(yre))/Lpx];

    % Save xr,yr in Plates/.../DefineKymographs
    cd(savedir);
    mkdir(listmov(i).platename);
    cd(listmov(i).platename);
    mkdir('DefineKymographs');
    cd('DefineKymographs');
    save(strcat(listmov(i).name(3:end),'.mat'),'xr','yr','L','Lpx','xtmp','ytmp','A');

    % Save Figure - DefineKymographs
    c=degrade(size(xr,2),'yr');
    for j=1:size(xr,2);
        plot(xr(j),yr(j),'.','color',c{1,j});
    end
    colormap(gray);
    cd(expdir);
    cd(listmov(i).platename);
    mkdir('DefineKymographs');
    cd('DefineKymographs');
    saveas(h,strcat(listmov(i).name(3:end),'.png'));

end

% Update listmov

for i=1:size(listmov,2)

    cd(savedir);
    cd(listmov(i).platename);
    cd('DefineKymographs');
    load(strcat(listmov(i).name(3:end),'.mat'));

    % Update listmov
    listmov(i).xr=xr;
    listmov(i).yr=yr;
    listmov(i).xtmp=xtmp;
    listmov(i).ytmp=ytmp;
    listmov(i).A=A;
    listmov(i).Lkym_um=L;
    listmov(i).Lkym_px=Lpx;

end

% Dated version Save
cd(logdir);
k=size(timetxt,2);
tmptime=datetime;
timetxt{1,k+1}=tmptime;
timetxt{2,k+1}="DefineKymographs";
save(strcat(char(tmptime),'.mat'),'listmov','listplat','timetxt');

% Current Listings Save
cd(logdir);
save("CurrentListings.mat",'listmov','listplat','timetxt');


%% MAKE KYMOGRAPHS

% Load Current Listings
cd(logdir);
load('CurrentListings.mat');

% Make Kymographs
for i=1:size(listmov,2)
    % Go to image folder
    cd(listmov(i).dirname);
    cd('Img');
    % Initialize kym
    kym=[];
    % Create waitbar
    h=waitbar(0,strcat('... i=',num2str(i),'/',num2str(size(listmov,2)),' ...'));
    % Create kymograph
    for j=1:size(listmov(i).listim,1)
        waitbar(j/listmov(i).Nim);
        A=imread(listmov(i).listim(j).name);
        kym=[kym, diag(A(round(listmov(i).yr),round(listmov(i).xr)))];
    end
    close(h);
    listmov(i).kym=kym;
    % Plot kymograph
    figure
    imagesc(kym);
    colormap(gray);
    % Save tif file
    cd(expdir);
    cd(listmov(i).platename);
    mkdir('Kymographs');
    cd('Kymographs');
    imwrite(kym,strcat(listmov(i).name(3:end),'.tif'));
    % Save Figure file
    cd(expdir);
    cd(listmov(i).platename);
    mkdir('KymoFig');
    cd('KymoFig');
    xlabel('t(fps)');
    ylabel('x(px)');
    saveas(gcf,strcat(listmov(i).name(3:end),'.png'));
    % Save matfile
    cd(savedir);
    cd(listmov(i).platename);
    mkdir('Kymographs');
    cd('Kymographs');
    save(strcat(listmov(i).name(3:end),'.mat'),'kym');
    
end

% Dated version Save
cd(logdir);
k=size(timetxt,2);
tmptime=datetime;
timetxt{1,k+1}=tmptime;
timetxt{2,k+1}="MakeKymographs";
save(strcat(char(tmptime),'.mat'),'listmov','listplat','timetxt');

% Current Listings Save
cd(logdir);
save("CurrentListings.mat",'listmov','listplat','timetxt');
