%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AMF_Transport_ListFiles %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% AMF
%   Transport
%       ListFiles: from sequence of images
%                           to .mat (listfiles)
%                           to .avi (movie)
%                           to FirstImage & LastImage (.png)
% Version PMMH - PhB - 240128 (Myristate) - last update: 240319

%% INTIALIZE

% Cleaning
clear all


% Base folders
sourcedir='/home/philippe/Documents/AMF_Transport/Data/ToAnalyze';
logdir='/home/philippe/Documents/AMF_Transport/Data/AnalysisLogFiles';
expdir='/home/philippe/Documents/AMF_Transport/Exp/Plates';
savedir='/home/philippe/Documents/AMF_Transport/Data/Plates';
saveindir='/home/philippe/Documents/AMF_Transport/Data/Indiv';
movdir='/home/philippe/Documents/AMF_Transport/Movies';

%% LISTINGS

% Create listplat
cd(sourcedir);
listplat=dir("2*Plate*");

% Prepare logfiles
timetxt{1,1}=datetime;
timetxt{2,1}="Ini";

% List Movies - listmov %

k=0;
for i=1:size(listplat,1)
    cd(sourcedir);
    cd(listplat(i).name);
    listmp=dir('0*');
    listdir=listmp([listmp.isdir]==1);
    listplat(i).listdir=listdir;
    for j=1:size(listmp)
        k=k+1;
        listmov(k).dirname=strcat(sourcedir,'/',listplat(i).name,'/',listdir(j).name);
        listmov(k).name=slash2_(strcat(listplat(i).name,'/',listdir(j).name));
        listmov(k).platename=listplat(i).name(3:end);
        listmov(k).param=GetParam(listmov(k).name);
        listmov(k).date=TrouvParam(listmov(k).param,'Date');
        listmov(k).plate=TrouvParam(listmov(k).param,'Plate');
        listmov(k).movnum=str2double(listmp(j).name);

        cd(listmov(k).dirname);
        clear Plate PrincePos root strain medium split Pbait CrossDate Temperature conv_px2mum fps t_crossing;
        %load('param.m')
        cd('Img');
        listim=dir('Img*.tif');
%         if isempty(listim)
%             listim=dir('*.bmp');
%         end
        listmov(k).Nim=size(listim,1);
        listmov(k).listim=listim;

    end
end

% Save Listings %

% Dated Save
cd(logdir);
k=size(timetxt,2);
tmptime=datetime;
timetxt{1,k+1}=tmptime;
timetxt{2,k+1}="Listings";
save(strcat(char(tmptime),'.mat'),'listmov','listplat','timetxt');

% Current Listings Save
cd(logdir);
save("CurrentListings.mat",'listmov','listplat','timetxt');

% %% ListFiles & Params
% cd(sourcedir);
% for i=1:size(listmov,2)
%     clear Plate PrincePos root strain medium split Pbait CrossDate Temperature conv_px2mum fps t_crossing
%     cd(sourcedir);
%     cd(listdir(i).name);
%     listmov(i).name=listdir(i).name;
%     
%     %cd('Img');
%     listim=dir('*.tiff');
%     if isempty(listim)
%         listim=dir('*.bmp');
%     end
%     listmov(i).Nim=size(listim,1);
%     
%     seqf=[];
%     for j=1:size(listim,1)
%         seqf=[seqf;str2double(listim(j).name(41:49))];
%     end
%     sequ=unique(seqf);
%     listmov(i).sequ=sequ;
%     listmov(i).nseq=size(sequ,1);
%     
%     nimpseq=[];
%     for k=1:size(sequ,1)
%         idx=[];
%         idxseq=find(seqf==sequ(k));
%         tmp=listim(idxseq);
%         for j=1:size(tmp,1)
%             idx=[idx;str2double(tmp(j).name(51:end-4))];
%         end
%         [tmpidx1,idxs]=sort(idx);
%         listseq=tmp(idxs);
%         cd(saveindir);
%         save(strcat(listmov(i).name,'_seq',num2str(k),'.mat'),'listseq');
%         nimpseq=[nimpseq size(idx,1)];
%     end
%     listmov(i).nimpseq=nimpseq;
%     
%     
%     cd(sourcedir);
%     cd(listdir(i).name);
%     if ~isempty(dir('param.m'))
%         run('param.m');
%         listmov(i).plate=Plate;
%         listmov(i).princepos=PrincePos;
%         listmov(i).root=root;
%         listmov(i).strain=strain;
%         listmov(i).medium=medium;
%         listmov(i).split=split;
%         listmov(i).pbait=Pbait;
%         listmov(i).crossdate=CrossDate;
%         listmov(i).temperature=Temperature;
%         listmov(i).px2um=conv_px2mum;
%         listmov(i).fps=fps;
%         listmov(i).tcrossing=t_crossing;
%         listmov(i).duration=listmov(i).Nim/listmov(i).fps; %s
%     else
%         listmov(i).px2um=3.45/25;
%         listmov(i).fps=20;
%         listmov(i).duration=listmov(i).Nim/listmov(i).fps; %s
%     end
%     
%     
%     
%     
% end
% 
% cd(savedir);
% save('listings.mat','listmov');

%% First and last Image

% Load Current Listings
cd(logdir);
load('CurrentListings.mat');

% Extract First & Last Image
for i=1:size(listmov,2)

    % Disp Name Movie
    listmov(i).name

    % Create Plate folder - exp to be saved
    cd(expdir);
    mkdir(listmov(i).platename);
    cd(listmov(i).platename);
    mkdir('FirstImage');
    mkdir('LastImage');

    % Commented - assuming listim only one sequence
%     for k=1:liscd(listmov(i).name);tmov(i).Nim
% 
%         cd(expdir);cd(listmov(i).name);
% 
%         cd(saveindir);
%         clear listseq;
%         load(strcat(listmov(i).name,'_seq',num2str(k),'.mat'));
        
    cd(listmov(i).dirname);
    cd('Img');
    
    A=imread(listmov(i).listim(1).name);
    B=imread(listmov(i).listim(end).name);
    
    cd(strcat(expdir,'/',listmov(i).platename,'/FirstImage'));
    imwrite(A,strcat(listmov(i).name,'.png'));
 
    cd(strcat(expdir,'/',listmov(i).platename,'/LastImage'));
    imwrite(B,strcat(listmov(i).name,'.png'));
    
    %end
end

% Dated version Save
cd(logdir);
k=size(timetxt,2);
tmptime=datetime;
timetxt{1,k+1}=tmptime;
timetxt{2,k+1}="FirstLast";
save(strcat(char(tmptime),'.mat'),'listmov','listplat','timetxt');

% Current Listings Save
cd(logdir);
save("CurrentListings.mat",'listmov','listplat','timetxt');

%% MAKE MOVIES


% Load Current Listings
cd(logdir);
load('CurrentListings.mat');

% Make Movies
for i=1:size(listmov,2);
    % Comment - assuming listim well sorted (only 1 seq + correct order of images)
%     for k=1:listmov(i).nseq
%         cd(saveindir);
%         clear listseq;
%         load(strcat(listmov(i).name,'_seq',num2str(k),'.mat'));
%         if i==1146
%             listseq=listseq(1:388);
%             save(strcat(listmov(i).name,'_seq',num2str(k),'.mat'),'listseq');
%         elseif i==1180
%             listseq=listseq(1:1146);
%             save(strcat(listmov(i).name,'_seq',num2str(k),'.mat'),'listseq');
%         end

    % Create Waitbar
    h = waitbar(0,strcat('... Please wait... Movie =',num2str(i),'/',num2str(size(listmov,2))));
    % Create mov
    cd(movdir);
    v = VideoWriter(strcat(listmov(i).name,'.avi'),'Motion JPEG AVI');
%     v = VideoWriter(strcat(listmov(1).name,'.mp4'),'MPEG-4');
%     v = VideoWriter(strcat(listmov(i).name,'_seq',num2str(k),'.avi'),'Motion JPEG AVI');
%     v.VideoFormat='gray';
%     v.FrameRate=listmov(i).fps;
%     v.Quality=80;
    v.FrameRate=20;
    open(v);   
    for j=1:size(listmov(i).listim,1)
        cd(strcat(listmov(i).dirname,'/Img'));
        A=imread(listmov(i).listim(j).name);
        colormap(gray);
        A2 = imresize(A,0.25);
        writeVideo(v,A2);
        waitbar(j/listmov(i).Nim,h)
    end
    close(v);
    close(h);
end

% Dated version Save
cd(logdir);
k=size(timetxt,2);
tmptime=datetime;
timetxt{1,k+1}=tmptime;
timetxt{2,k+1}="Movies";
save(strcat(char(tmptime),'.mat'),'listmov','listplat','timetxt');

% Current Listings Save
cd(logdir);
save("CurrentListings.mat",'listmov','listplat','timetxt');


