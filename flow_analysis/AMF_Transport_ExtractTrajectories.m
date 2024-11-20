%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AMF_Transport_ExtractTrajectories %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% AMF
%   Transport
%       ExtractTrajectories: from .mat (individual files - ...exp/Plates/.../AnalysisKymographs)
%                           to .mat (individual files - ...exp/Plates/.../Trajectories)
%                           to .png (individual files - ...exp/Plates/.../KymoTrajs)
%       ComputeVelocities: from Trajectories to Velocities
%                           to .mat (individual files - ...exp/Plates/.../Trajectories)
%                           to .csv (individual files - export)
%                           to .png (individual files - ...exp/Plates/.../Velocities)
%       PlateVelocities: from individual movies to all movies on a plate
%                           to .mat
%                           to .png
% Version PMMH - 240207 (Myristate) - lats update: 240319

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
exportdir='/home/philippe/Documents/AMF_Transport/Export/AllVelocities';
figdir='/home/philippe/Documents/AMF_Transport/Figures/Plates';

%% EXTRACT TRAJECTORIES

% Load Current Listings
cd(logdir);
load('CurrentListings.mat');

% Pick parameters
L=20; %um
px2um=3.45/50*2; % Magnification x50
fps=20;
% ChoicePlate=find([listseq.plate]==408);

for i=1:size(listmov,2)
    
    % Initialize
    %close all

    % Disp platename
    listmov(i).name

    % Initialize antrajs
    clear antrajs trajs V;
    antrajs=[];

    % Load AnalysisKymographs
    cd(strcat(expdir,'/',listmov(i).platename,'/AnalysisKymographs/Split_N1')); %Split_N1 -> L= 40um /Split_N2
    cd(listmov(i).name(3:end));
    clear Expression* trajs;
    testfile=dir('Part1.mat');
    load('Part1.mat');

    % Extract trajs from .mat
    if testfile.bytes<=128 || isempty(Expression1)
            rawtraj=[];
            trajs=[];
            ntrajs=0;
        elseif size(size(Expression1),2)==2
            rawtraj=Expression1;
            ntrajs=size(Expression1,1);
            for k=1:size(Expression1,1)
                rawtrajs=Expression1;
                trajs(k).vec=double(Expression1{k,1});
                trajs(k).tpx=trajs(k).vec(:,1);
                trajs(k).xpx=trajs(k).vec(:,2);
            end
        elseif size(size(Expression1),2)==3 & size(Expression1,1)>1
            rawtraj=Expression1;
            ntrajs=size(Expression1,3);
            for k=1:size(Expression1,3)
                trajs(k).vec=double(Expression1(:,:,k)');
                trajs(k).tpx=trajs(k).vec(:,1);
                trajs(k).xpx=trajs(k).vec(:,2);
            end
        elseif size(size(Expression1),2)==3 & size(Expression1,1)==1
            rawtraj=Expression1;
            ntrajs=1;
            trajs(k).vec=[Expression1(:,:,1)' Expression1(:,:,2)'];
            trajs(k).tpx=trajs(1).vec(:,1);
            trajs(k).xpx=trajs(1).vec(:,2);
            
    end

    % ComputeVelocities on trajs
    for k=1:size(trajs,2)
        trajs(k).tpx=size(listmov(i).kym,2)-trajs(k).tpx;
        trajs(k).ts=trajs(k).tpx./fps;
        trajs(k).xum=trajs(k).xpx.*px2um;
        [p,e]=phitpoly1(trajs(k).ts,trajs(k).xum);
        trajs(k).t0=min(trajs(k).ts);
        trajs(k).dur=max(trajs(k).ts)-min(trajs(k).ts);
        trajs(k).V=p(1);
        trajs(k).x0=p(2);
        trajs(k).errfit=e;
        clear p e;
    end

    % Create V
    V=[trajs.V];
    t0=[trajs.t0];
    x0=[trajs.x0];
    dur=[trajs.dur];
    errfit=[trajs.errfit];

    % Save trajs (.mat)
    cd(expdir);
    cd(listmov(i).platename);
    mkdir('Trajectories');
    cd('Trajectories');
    save(strcat(listmov(i).name(3:end),'.mat'),'trajs','V');

    % Export V (.csv)
    cd(exportdir);
    csvwrite(strcat(listmov(i).name(3:end),'.csv'),V);

    % Plot Trajectories
    cd(strcat(expdir,'/',listmov(i).platename));
    mkdir('KymoTrajs');
    cd('KymoTrajs');
    % Plot Kymo
    figure
    imagesc(listmov(i).kym);
    colormap(gray);
    hold on
    % Plot all trajs
    for k=1:size(trajs,2)
        if V(k)>0
            c=[1 0 0];
        else
            c=[0 0 1];
        end
        plot(trajs(k).tpx,trajs(k).xpx,'color',c);
        hold on
    end
    xlabel('t(frames)');
    ylabel('x(px)');
    saveas(gcf,strcat(listmov(i).name(3:end),'.png'));

    % Plot Velocities
    cd(strcat(expdir,'/',listmov(i).platename));
    mkdir('Velocities');
    cd('Velocities');
    figure
    plot(t0(V>0),V(V>0),'^r');
    hold on
    plot(t0(V<=0),V(V<=0),'vb');
    xlabel('t0(s)');
    ylabel('V(um/s)');
    saveas(gcf,strcat(listmov(i).name(3:end),'.png'));

    % Update listmov
    listmov(i).V=V;
    listmov(i).t0=t0;
    listmov(i).dur=dur;
    listmov(i).Vp=V(V>=0);
    listmov(i).Vn=V(V<=0);
    listmov(i).mVp=mean(listmov(i).Vp);
    listmov(i).mVn=mean(listmov(i).Vn);
    listmov(i).eVp=std(listmov(i).Vp);
    listmov(i).eVn=std(listmov(i).Vn);

end

% Dated version Save
cd(logdir);
k=size(timetxt,2);
tmptime=datetime;
timetxt{1,k+1}=tmptime;
timetxt{2,k+1}="ComputeVelocities";
save(strcat(char(tmptime),'.mat'),'listmov','listplat','timetxt');

% Current Listings Save
cd(logdir);
save("CurrentListings.mat",'listmov','listplat','timetxt');



% %% PLATE VELOCITIES
% 
% % Load Current Listings
% cd(logdir);
% load('CurrentListings.mat');
% 
% % Group V by plates
% %platenam='230515_Plate059';
% %tmpidx=[1:8];
% %platenam='230518_Plate049';
% %tmpidx=[9:17];
% platenam='230524_Plate056';
% tmpidx=[18:29];
% platenam='230524_Plate056';
% tmpidx=[18:29];
% allV=[];
% for i=tmpidx;%1:size(listmov,2)
%     %if size(listmov(i).platename)==size(platenam) && sum(listmov(i).platename==platenam)
%         %tmpidx=[tmpidx i];
%         allV=[allV listmov(i).V];
%     %end
% end
% 
% % Save allV
% cd(strcat(expdir,'/',platenam));
% save('AllV.mat','allV');
% 
% % Plot allV
% figure
% hist(allV,linspace(-10,10,100));
% xlabel('V (um/s)');
% ylabel('N');
% saveas(gcf,strcat(platenam,'_AllV.png'))
% 
% 
% 
% 
% 
% 









