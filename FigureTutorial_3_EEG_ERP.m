%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                  EXAMPLE 3: ERP PLOT - HACKING EEGLAB                  %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The following code uses data provided by Mike X. Cohen. Data here:
% 
% http://mikexcohen.com/book/AnalyzingNeuralTimeSeriesData_MatlabCode.zip
%
% This code uses EEGLAB to create and ERP plot. Then we hack the plot to
% make it look better. EEGLAB can be downloaded from their website:
%
% https://sccn.ucsd.edu/eeglab/index.php
%
% Code written by: Justin Brantley
% email: justin dot a dot brantley at gmail dot com

clc
clear
close all;

% Sample EEG from Mike X. Cohen
load(fullfile('data','sampleEEGdata.mat'));
EEG.data = double(EEG.data);

% Load colors - to be used later
bc = blindcolors;
rr = [228,27,29]./255;  % red
bb = [59,129,185]./255; % blue

% Define figure
ff = figure('color','w','units','inches','position',[2,2 8 8]);

% Call default erp plot in eeglab
plottopo(mean(EEG.data,3),EEG.chanlocs);

%% Define channels to plot

% chans2plot = {'Fp1','Fp2','AFz','F5','F1','F2','F4','FC3','FCz','FC4',...
%               'C5','C1','C2','C6','CP3','CPz','Cp4','P5','P1','P2','P6','PO3','POz','PO4','Oz'};
chans2plot = {'AFz','F3','F4','Fz','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8',...
    'CP5','CP1','CP2','CP6','P8','P4','Pz','P3','P7','PO7','POz','PO8','Oz'};
% Get channel numbers
channums = cellfun(@(x) find(strcmpi({EEG.chanlocs.labels},x)),chans2plot);
% Define figure
tempfig = figure('color','w','units','inches','position',[2,2 8 8]);
% Call default erp plot in eeglab
plottopo(mean(EEG.data(channums,:,:),3),EEG.chanlocs(channums));

%% Start making it pretty - Build skeleton of figure

% Create new figure
fig2 = figure('color','w','units','inches','position',[2,2 8 8]);

% Get axis position for each small plot in default plottopo
axpos = flipud(cat(1,tempfig.Children(2:end-1).Position));

% Pre allocate variable for new axes
axall = cell(1,length(chans2plot));

% Define window of data to plot
win = [-200 700];
xvals = dsearchn(EEG.times',win(:));

% Start generating each one
for ii = 1:length(chans2plot)
    % Get channel number
    channum = find(strcmpi({EEG.chanlocs.labels},chans2plot(ii)));
    
    % Create axis
    ax = axes;
    ax.Units = 'normalized';
    
    % Define position based on plottopo
    ax.Position(1:end) = axpos(ii,:);
    ax.Position(1) = ax.Position(1) - .05;
    ax.Position(2) = ax.Position(2) - .05;
    ax.Position(3) = .1;
    ax.Position(4) = .06;
    
    % Plot the data
    p1 = plot(EEG.times(xvals(1):xvals(2)),mean(EEG.data(channum,xvals(1):xvals(2),:),3),'color',bb,'linewidth',1.5);
    
    % Add axis to variable
    axall{ii} = ax;
end

%% Add shaded error bars

% Start generating each one
for ii = 1:length(chans2plot)
    
    % Get channel number
    channum = find(strcmpi({EEG.chanlocs.labels},chans2plot(ii)));
    % Activate axis
    axes(axall{ii});
   
    % Add errorbars
    p1 = shadedErrorBar(EEG.times(xvals(1):xvals(2)),mean(EEG.data(channum,xvals(1):xvals(2),:),3),std(EEG.data(channum,xvals(1):xvals(2),:),[],3),{'markerfacecolor',bb});
    
    % Clean up
    p1.mainLine.Color = bb;
    delete(p1.edge(1:2));
    p1.patch.FaceAlpha = 0.5;
end

%% Begin editing axes

% Set y limits
y_limits = [-20 20];

% Edit each one
for ii = 1:length(chans2plot)
    
    % Get channel number
    channum = find(strcmpi({EEG.chanlocs.labels},chans2plot(ii)));
    % Activate axis
    axes(axall{ii});
   
    % Turn off box
    axall{ii}.Box = 'off';
    
    % We dont need ticks here 
    axall{ii}.XTick = [];
    axall{ii}.YTick = [];
    
    % Put x-axis at origin
    axall{ii}.XAxisLocation = 'origin';
    axall{ii}.YAxisLocation = 'origin';
    
    % Set y limits
    ylim(y_limits)
    
end

%% Add labels and clean up

% Start generating each one
for ii = 1:length(chans2plot)
    
    % Get channel number
    channum = find(strcmpi({EEG.chanlocs.labels},chans2plot(ii)));
    % Activate axis
    axes(axall{ii});
   
    % Add y labels
    axall{ii}.YLabel.String = chans2plot{ii};
    axall{ii}.YLabel.Rotation = 0;
    axall{ii}.YLabel.HorizontalAlignment = 'right';
    axall{ii}.YLabel.VerticalAlignment = 'middle';
    axall{ii}.YLabel.FontSize = 12;
    axall{ii}.YLabel.Position(1) = EEG.times(xvals(1)) - 50;
    axall{ii}.YLabel.Position(2) = 7.5;
    
    % Beef up the lines
    axall{ii}.YAxis.LineWidth = 1.15;
    axall{ii}.XAxis.LineWidth = 1.15;

end

%% Now the axes look decent. Lets add a scale bar to know what we are looking at

% Create axis
axkey = axes;
axkey.Position = [.125 .75 .15 .1];
axkey.Box = 'off';
axkey.FontSize = 10;
% Set limits
axkey.XLim = win;
axkey.YLim = y_limits;

% Define axis color
axkey.XColor = 'k';
axkey.YColor = 'k';

% Set ticks
axkey.YTick = y_limits;
axkey.XTick = [250 500];
axkey.XLabel.String = '(ms)';
axkey.XLabel.Position = [375 -11 0];
axkey.XLabel.HorizontalAlignment = 'center';
axkey.XLabel.VerticalAlignment = 'top';

% Add t = 0 marker
txt1 = text(0, -25,'t = 0','HorizontalAlignment','center','VerticalAlignment','top','FontSize',10);

% % Set tick direction and length
% axkey.TickDir = 'out';
% axkey.TickLength(2) = .5;

% Set axis location to origin
axkey.XAxisLocation = 'origin';
axkey.YAxisLocation = 'origin';

% Manually add y-labels since setting Y Axis to origin makes them disappear 
ylab1 = text(EEG.times(dsearchn(EEG.times',-50)), 18,'20','HorizontalAlignment','right');
ylab2 = text(EEG.times(dsearchn(EEG.times',-50)), -18,'-20','HorizontalAlignment','right');

% Label y value
axkey.YLabel.String = '\muV';
axkey.YLabel.Rotation = 0;
axkey.YLabel.HorizontalAlignment = 'right';
axkey.YLabel.VerticalAlignment = 'middle';
axkey.YLabel.Position(1) = EEG.times(xvals(1)) - 50;
axkey.YLabel.Position(2) = 0;

% Beef up linewidth
axkey.YAxis.LineWidth = 1.15;
axkey.XAxis.LineWidth = 1.15;

% % Add x scale bar
% xscale = line([EEG.times(dsearchn(EEG.times',350)) EEG.times(dsearchn(EEG.times',650))],[-7.5 -7.5],'color','k','linewidth',1);
% txt2 = text(EEG.times(dsearchn(EEG.times',225)), -12.5,'300 ms','HorizontalAlignment','left','FontSize',10);

% Add line and patch showing mean and std
l_mean = line([250 500],[18 18],'color',bb,'linewidth',2);
txt_mean = text(550,20,'\mu','FontSize',12);
xx = [250 500 500 250];
yy = [6 6 12 12];
p_std  = patch(xx,yy,bb,'FaceAlpha',0.5,'EdgeColor','none');
txt_std = text(550,11,'\sigma','FontSize',12);

% Add arrow. Note: annotation uses figure coordinates not axis coordinates.
% Use the function ds2nfu (in matlab central) to convert axis to fig coords
% [xaf,yaf] = ds2nfu([250 0],[40 22]); annotation('arrow',xaf,yaf);
ann = annotation('textarrow',[.2 .1585],[.9 .855]);
ann.HeadWidth = 8;
axkey.FontSize = 12;
txt_ann = text(300,40,'Extremely Painful Stimulus!','FontSize',12,'FontWeight','n','FontAngle','italic');
txt_ann.FontSize =12;

% Fix fontsize except added text for line labels
axkey.FontSize = 10;

%% Add topoplot to show electrode locations on scalp
% 
% axtopo = axes;
% axtopo.Position = [.625 .625 .4 .4];
% %hold on;
% topoplot([],EEG.chanlocs(channums),'electrodes','labels','headrad',.5,'plotrad',.65,'whitebk','on');
% axtopo.Children(end-1).LineWidth = 1.15;
% axtopo.Children(end).FaceColor = 'w';

axtopo = axes;
%topoplot([],EEGHI.chanlocs,'style','blank','electrodes','off');
hold on;
topoplot([],EEG.chanlocs(channums),'electrodes','labels','headrad',.5,'plotrad',.65);
axtopo.Children(end-6).Visible = 'off';
axtopo.Children(end-5).Visible = 'off';
axtopo.Children(end-1).LineWidth = 1.15;
axtopo.Position = [.645 .645 .35 .35];

% Fix figure color
fig2.Color = 'w';

