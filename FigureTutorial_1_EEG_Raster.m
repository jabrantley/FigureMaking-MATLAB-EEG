%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                  EXAMPLE 1: EEG TIMESERIES + ANIMATION                 %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Written by: Justin Brantley
% email: justin dot a dot brantley at gmail dot com
clc
clear
close all;

% Add paths
addpath(genpath('utils'));
addpath('eeglab2019_1');
eeglab; clc; clear; close all;

% Load data
load(fullfile(cd,'data','motionTest'));

% Convert to double and transpose. Now data are: [channels x time]
eeg = double(eeg);
eog = double(eog);
fs  = 1000; % Hz

% Run Hinfinity
% eye_artifacts = [eog(:,4) - eog(:,3), eog(:,2) - eog(:,1), ones(size(eeg,1),1)];
% eeg_hinf = transpose(hinfinity(eeg,eye_artifacts));

% HP Filter for visualization
eeg = transpose(filterdata('data',eeg,'srate',fs,'highpass',0.3,'highorder',2));
eog = transpose(filterdata('data',eog,'srate',fs,'highpass',0.3,'highorder',2));

% Choose channels to plot: Fp1, Fz, Cz, Pz, Oz
chans2plot = [2, 5, 14, 23, 27];

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    %
%       Plot HP filt EEG             %
%                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define figure window
fig1 = figure('color','w','units','inches','position',[1, 4, 7.5, 3.5]);

% Get axes
ax = gca;
ax.Box = 'on';

% Define offset for each timeseries
offset = 150;

% Define window to plot
win = eyesOpenIDX(1):eyesOpenIDX(2);

% Initialize empty varaible for each plot
pp = [];

% Plot each EEG timeseries
hold on;
for ii = 1:length(chans2plot)
    % Get channel index
    idx = chans2plot(ii);
    % Offset data
    data = eeg(idx,win) - (ii-1)*offset;
    % Plot data
    pp(ii) = plot(data,'k');
    
end

%% Make minor adjustments to figure
% Y limits
ylim([-750 200])
% X limits
xlim([0 length(win)])
% Change x and y label
xlabel('Time (sec)')
ylabel('EEG (\muV)')

%export_fig eeg_raster_1.png -png -r300

%% Make new figure centered around eyes open/closed
delete(fig1)
fig1 = figure('color','w','units','inches','position',[1, 4, 7.5, 3.5]);

% Get axes
ax = gca;
ax.Box = 'on';

% Define offset for each timeseries
offset = 150;

% Define window to plot
tt = 5*fs;
win = (eyesCloseIDX(1)-tt) : (eyesCloseIDX(1)+tt);

% Initialize empty varaible for each plot
all_plot = [];

% Plot each EEG timeseries
hold on;
for ii = 1:length(chans2plot)
    % Get channel index
    idx = chans2plot(ii);
    % Offset data
    data = eeg(idx,win) - (ii-1)*offset;
    % Plot data
    pp = plot(data,'k');
    % Add plot to all plot variable
    all_plot = [all_plot; pp];
    
end

% Y limits
ylim([-750 200])
% X limits
xlim([0 length(win)])
% Change x and y label
xlabel('Time (sec)')
ylabel('EEG (\muV)')

%% Add line to identify eyes open ---> closed
ll = line([5000 5000],[-750 200],'linestyle','--','color',0.5.*ones(1,3));

%% Change colors of EEG to identify channel names

% Get colorblind colors
colors = blindcolors;
color_ord = [2,3,4,8,6];

% Change each line color
for ii = 1:length(chans2plot)
    all_plot(ii).Color = colors(color_ord(ii),:);
end

% Add legend
leg = legend(all_plot,{'Fp2','Fz','Cz','Pz','Oz'});

%export_fig eeg_raster_2.png -png -r300
%% Changing colors isnt so great - Lets label each of the time series

% Delete the legend
delete(leg);

% Update y ticks based on offset
ax.YTick = fliplr(0 - ((1:length(chans2plot))-1)*150);

% Update y tick labels
ax.YTickLabel = fliplr({'Fp2','Fz','Cz','Pz','Oz'});

% Update x ticks
ax.XTick = (0:10)*fs;
ax.XTickLabel = sprintfc('%d',0:10);

% Add title
title('Eyes Open vs Eyes Closed');

% Change color
for ii = 1:length(chans2plot)
    all_plot(ii).Color = 'k';
end

%% X Ticks should be referenced to event
ax.XTickLabel = sprintfc('%d',-5:5);

%export_fig eeg_raster_3.png -png -r300
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    %
%    Filter data into alpha band     %
%                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eeg_alpha = transpose(filterdata('data',eeg','srate',fs,...
    'highpass',8,'highorder',2,...
    'lowpass',12,'loworder',2));

%% Plot alpha data
delete(fig1)
fig1 = figure('color','w','units','inches','position',[1, 4, 7.5, 3.5]);

% Get axes
ax = gca;
ax.Box      = 'on';

% Define offset for each timeseries
offset = 15;

% Define window to plot
t = 5;
tt = t*fs;
win = (eyesCloseIDX(1)-tt) : (eyesCloseIDX(1)+tt);

% Initialize empty varaible for each plot
all_plot = [];

% Plot each EEG timeseries
hold on;
for ii = 1:length(chans2plot)
    % Get channel index
    idx = chans2plot(ii);
    % Offset data
    data = eeg_alpha(idx,win) - (ii-1)*offset;
    % Plot data
    pp = plot(data,'k','linewidth',1.1);
    % Add plot to all plot variable
    all_plot = [all_plot; pp];
    
end

% Y limits
ylim([-offset*length(chans2plot)+offset/2 offset])
% X limits
xlim([0 length(win)])
% Change x and y label
xlabel('Time (sec)')
ylabel('EEG (\muV)')

% Add line to show eyes open vs eyes closed
ll = line([tt tt],ylim,'linestyle','--','color',0.5.*ones(1,3),'linewidth',1.5);

% Update y ticks based on offset
ax.YTick = fliplr(0 - ((1:length(chans2plot))-1)*offset);

% Update y tick labels
ax.YTickLabel = fliplr({'Fp1','Fz','Cz','Pz','Oz'});

% Update x ticks
ax.XTick = (0:2*t)*fs;
ax.XTickLabel = sprintfc('%d',-t:t);

% Add title
title('Eyes Open vs Eyes Closed');

% export_fig eeg_raster_4.png -png -r300

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    %
%           Add some style!          %
%                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Turn off box
ax.Box = 'off';

% Change axis line color to none (can be changed to anything)
ax.XColor = 'none';
ax.YColor = 'none';

% Delete title
ax.Title.String = [];

% Add ax labels back in manually
my_labels = [];
chan_labels = {'Fp2','Fz','Cz','Pz','Oz'};
for ii = 1:length(chans2plot)
    temp = text(-600, -(ii-1)*offset,chan_labels(ii),'fontsize',11,'fontweight','b');
    my_labels = [my_labels; temp];
end

% Add scaling - x axis
xscale = line([0, 1*fs]+100,-offset*length(chans2plot)*ones(1,2)+(offset/2),'color','k','linewidth',1.5);
xscale_text = text(100,-offset*length(chans2plot)+(offset/3),'1 second');

% Add scaling - y axis
xlim([0 length(data)+100])
yscale = line(length(data)+100*ones(1,2), [-5,5] - 2*offset,'color','k','linewidth',1);
yscale_text = text(length(data)+200, - 2*offset, '10 \muV');

% Add labels for eyes open and closed
label_open = text(2000,3*offset/4,'Eyes Open','fontweight','b','fontsize',12);
label_close = text(7000,3*offset/4,'Eyes Closed','fontweight','b','fontsize',12);

%% Highlight certain part of timeseries

% Initialize NaN vectors for each channel
highlight_pz = nan(1,length(data));
highlight_oz = nan(1,length(data));

% Fill in data to plot in different color
highlight_win = 6*fs : 9*fs;
highlight_pz(highlight_win) = all_plot(4).YData(highlight_win);
highlight_oz(highlight_win) = all_plot(5).YData(highlight_win);

% Plot higlight data
p_pz = plot(highlight_pz,'color',colors(8,:),'linewidth',1.1);
p_oz = plot(highlight_oz,'color',colors(6,:),'linewidth',1.1);

% Remove white space to fill figure size
ax.Position = [.075 .05 .85 .9];

%% Now lets export the figure to png using export_fig

% Export figure (uses print function underneath but performs much better)
export_fig eeg_raster_final_V1.png -png -r300

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                         Alpha Plot V2                                  %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Define figure
fig2 = figure('color','w','units','inches','position',[1, 4, 7.5, 3.5]);
% Create axes
% There's a negative value BELOW so that the axes overlap slightly
ax = tight_subplot(5, 1, [-.025 .01] , [.05 .115], [.085 .085]);
for ii = 1:length(ax)
    ax(ii).Box = 'on';
    ax(ii).YTick = 0;
    ax(ii).YLim = [-1 1]
end
% Add axis behind these data axes
ax_pos = [ax(end).Position(1),...
    ax(end).Position(2),...
    ax(end).Position(3),...
    ax(1).Position(2) + ax(1).Position(4)];% - ax(end).Position(2)]; % <-- This would make the top ax_back edge match the top of ax(1)

ax_back = axes('position',ax_pos,'color','r','clipping','off','Box','on','YTick',[]);
% Move axis to the bottom
uistack(ax_back,'bottom')
ax_open = axes('color','b','position',[.2, ax(1).Position(2) + ax(1).Position(4) - ax(end).Position(2)/2, .2, .01],'box','on');
ax_closed = axes('color','b','position',[.6, ax(1).Position(2) + ax(1).Position(4) - ax(end).Position(2)/2, .2, .01],'box','on');

export_fig eeg_raster_template_V2.png -png -r300
%% Define figure
% close(fig2);
fig2 = figure('color','w','units','inches','position',[1, 4, 7.5, 3.5]);
% Create axes - there's a negative value BELOW so that the axes overlap slightly
ax = tight_subplot(5, 1, [-.025 .01] , [.05 .115], [.085 .085]);
% Channel labels
chan_labels = {'Fp2','Fz','Cz','Pz','Oz'};
% Get all data to compute limits
alpha_data = eeg_alpha(chans2plot,win);
minVal = min(alpha_data(:));
maxVal = max(alpha_data(:));
limVal = max(abs([minVal,maxVal]));

% Loop through data and add to each axis
for ii = 1:length(chans2plot)
    % Change which axis is active
    axes(ax(ii));
    % Get channel index
    idx = chans2plot(ii);
    % Offset data
    data = eeg_alpha(idx,win); % No need for offest now: - (ii-1)*offset;
    % Plot data
    pp = plot(data,'k','linewidth',1.1);
    % Change x limits
    ax(ii).XLim = [0 length(win)];
    
    % Change y limits
    % ax(ii).YLim = [-7.5 7.5];
    ax(ii).YLim = [-limVal limVal];
    
    % % --------------------------- %
    % % Note: This only works if you dont change the color to 'none'.
    % % Otherwise it also takes the color 'none'.
    % % Add ylabel
    % ax(ii).YLabel.String = chan_labels{ii};
    % % Rotate to 0-deg to make it L/R instead of up/down
    % ax(ii).YLabel.Rotation = 0;
    % % --------------------------- %
    
    % Turn off axis lines
    ax(ii).XColor = 'none';
    ax(ii).YColor = 'none';
    
    % Turn off clipping
    ax(ii).Clipping = 'off';
    % Turn off background color so some lines arent cut off
    ax(ii).Color = 'none';
    
    % Add ax label - add manually using text since we turned off the axis
    % NOTE: the field UserData is empty ([] ) and we can add anything we
    % want to it. Here, we will add a custom y label using text
    ax(ii).UserData.my_ylabel = text(-600, 0, chan_labels(ii),'fontsize',11,'fontweight','b');
    
end

% Add axis behind these data axes
ax_pos = [ax(end).Position(1),...
    ax(end).Position(2),...
    ax(end).Position(3),...
    ax(1).Position(2) + ax(1).Position(4)];% - ax(end).Position(2)]; % <-- This would make the top ax_back edge match the top of ax(1)

ax_back = axes('position',ax_pos,'color','none','clipping','off');

% Update background axis limits
ax_back.XLim = [0 length(win)];
%ax_back.YLim = [-.15 1];

% Turn off axis lines
ax_back.XColor = 'none';
ax_back.YColor = 'none';

% Move axis to the bottom
% uistack(ax_back,'bottom')

% Add line to show eyes open vs eyes closed
ll = line([tt tt],ylim,'linestyle','--','color',0.5.*ones(1,3),'linewidth',1.5);

% Add scaling - x axis
xscale = line([0, 1*fs]+100,[0 0],'color','k','linewidth',1.5);
xscale_text = text(100,-.03,'1 second');

% Add scaling - y axis
axes(ax(3));
yscale = line(length(data)+100*ones(1,2), [-5,5],'color','k','linewidth',1);
yscale_text = text(length(data)+200, 0, '20 \muV');

% Initialize NaN vectors for each channel
highlight_pz = nan(1,length(data));
highlight_oz = nan(1,length(data));
highlight_win = 6*fs : 9*fs;

% Plot higlight data - Pz
axes(ax(4)); hold on;
highlight_pz(highlight_win) = ax(4).Children(2).YData(highlight_win);
p_pz = plot(highlight_pz,'color',colors(8,:),'linewidth',1.1);

% Plot higlight data - Oz
axes(ax(5)); hold on;
highlight_oz(highlight_win) = ax(5).Children(2).YData(highlight_win);
p_oz = plot(highlight_oz,'color',colors(6,:),'linewidth',1.1);

% Add eyes open/closed labels using titles
ax_open = axes('color','w','position',[.2, ax(1).Position(2) + ax(1).Position(4) - ax(end).Position(2)/2, .2, .01]);
ax_open.Title.String    = 'Eyes Open';
ax_open.Title.FontSize  = 12;
ax_open.XColor = 'none';
ax_open.YColor = 'none';

ax_closed = axes('color','w','position',[.6, ax(1).Position(2) + ax(1).Position(4) - ax(end).Position(2)/2, .2, .01]);
ax_closed.Title.String    = 'Eyes Closed';
ax_closed.Title.FontSize  = 12;
ax_closed.XColor = 'none';
ax_closed.YColor = 'none';

%% Get fancy and add eye open/closed image
ax_imopen = axes('color','w','position',[.135, ax(1).Position(2) + ax(1).Position(4) - ax(end).Position(2)/2, .1, .1]);
I_open = imread(fullfile('misc','eye_open.png'));
imshow(I_open);

ax_imclosed = axes('color','w','position',[.535, ax(1).Position(2) + ax(1).Position(4) - ax(end).Position(2)/2, .1, .1]);
I_closed = imread(fullfile('misc','eye_closed.png'));
imshow(I_closed);

%%
export_fig eeg_raster_final_V2.png -png -r300
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                    Alpha Plot V3 - Lets animate!                       %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define figure
fig3 = figure('color','w','units','inches','position',[1, 4, 7.5, 3.5]);
% Create axes
% There's a negative value BELOW so that the axes overlap slightly
ax = tight_subplot(5, 1, [-.025 .01] , [.05 .115], [.085 .085]);
% Channel labels
chan_labels = {'Fp2','Fz','Cz','Pz','Oz'};

% Get window of data
alpha_data     = eeg_alpha(chans2plot,eyesOpenIDX(1):eyesOpenIDX(11));
all_alpha_data = eeg_alpha(:,eyesOpenIDX(1):eyesOpenIDX(11));

% Label eyes open/closed
eyes_open = zeros(1,size(alpha_data,2));
events    = [eyesOpenIDX(1:end-1)', eyesCloseIDX(:)] - eyesOpenIDX(1) + 1 ;
for ii = 1:size(events,1)
    eyes_open(events(ii,1):events(ii,2)) = 1;
end

%Compute limits
minVal = min(alpha_data(:));
maxVal = max(alpha_data(:));
limVal = max(abs([minVal,maxVal]));

% Define values for animation
win         = [1 10*fs];
update_rate = 1/25;
win_length  = 1*fs;
stat_win    = [1 win_length];
alpha_power = [];
all_alpha_power = [];

while stat_win(end) < size(alpha_data,2)
    alpha_win = sum(abs(alpha_data(:,stat_win(1):stat_win(2))).^2,2);
    alpha_power = [alpha_power, alpha_win]; % Not efficient!
    all_alpha_power = [all_alpha_power, sum(abs(all_alpha_data(:,stat_win(1):stat_win(2))).^2,2)];;
    stat_win = stat_win + fs*update_rate;
end

% Compute stats on data
mean_alpha = mean(alpha_power(:));
std_alpha  = std(alpha_power(:));
high_power = nan(size(alpha_power));
high_power(alpha_power >= mean_alpha + 3*std_alpha) = 1;

% Mask original data to highlight
alpha_highlight = nan(size(alpha_data));
stat_win    = [1 fs*update_rate];
cnt = 1;
while stat_win(end) < size(alpha_data,2)
    if high_power(cnt) == 1
        alpha_highlight(:,stat_win(1):stat_win(2)) = alpha_data(:,stat_win(1):stat_win(2));
    end
    stat_win = stat_win + fs*update_rate;
    cnt = cnt + 1;
end

% Loop through data and add to each axis
for ii = 1:size(alpha_data,1)
    % Change which axis is active
    axes(ax(ii));
    
    % Plot data
    pp = plot(alpha_data(ii,win(1):win(2)),'k','linewidth',1.1);
    hold on;
    hl = plot(alpha_highlight(ii,win(1):win(2)),'color',colors(color_ord(ii),:),'linewidth',1.1);
    
    % Change x limits
    ax(ii).XLim = win;
    
    % Change y limits
    % ax(ii).YLim = [-7.5 7.5];
    ax(ii).YLim = [-limVal limVal];
    
    % % --------------------------- %
    % % Note: This only works if you dont change the color to 'none'.
    % % Otherwise it also takes the color 'none'.
    % % Add ylabel
    % ax(ii).YLabel.String = chan_labels{ii};
    % % Rotate to 0-deg to make it L/R instead of up/down
    % ax(ii).YLabel.Rotation = 0;
    % % --------------------------- %
    
    % Turn off axis lines
    ax(ii).XColor = 'none';
    ax(ii).YColor = 'none';
    
    % Turn off clipping
    ax(ii).Clipping = 'off';
    % Turn off background color so some lines arent cut off
    ax(ii).Color = 'none';
    
    % Add ax label - add manually using text since we turned off the axis
    % NOTE: the field UserData is empty ([] ) and we can add anything we
    % want to it. Here, we will add a custom y label using text
    ax(ii).UserData.my_ylabel = text(-600, 0, chan_labels(ii),'fontsize',11,'fontweight','b');
    
end

% Add axis behind these data axes
ax_pos = [ax(end).Position(1),...
    ax(end).Position(2),...
    ax(end).Position(3),...
    ax(1).Position(2) + ax(1).Position(4) - ax(end).Position(2)]; % <-- This would make the top ax_back edge match the top of ax(1)

ax_back = axes('position',ax_pos,'color','none','clipping','off');
% For debugging eyes open/closed:
% peyes_open = plot(eyes_open(win(1):win(2)));

% Update background axis limits
ax_back.XLim = win;

% Turn off axis lines
ax_back.XColor = 'none';
ax_back.YColor = 'none';

% Add line to show eyes open vs eyes closed
%ll = line([round((win(1)+win(2))/2), round((win(1)+win(2))/2)],ylim,'linestyle','--','color',0.5.*ones(1,3),'linewidth',1.5);
axes(ax_back)
midpoint= round((win(1)+win(2))/2);
rr = rectangle('position',[midpoint-win_length/2 0 win_length 1],...
    'FaceColor', 0.9.*ones(3,1), 'LineStyle','none');

% Move axis to the bottom
uistack(ax_back,'bottom')

% Add scaling - x axis
%xscale = line([midpoint-win_length/2, midpoint+win_length/2],[0 0],'color','k','linewidth',1.5);
xscale_text = text(midpoint-win_length/2+50,-.03,'1 second','fontweight','b');

% Add scaling - y axis
axes(ax(end));
yscale = line(length(data)+100*ones(1,2), [-5,5],'color','k','linewidth',1);
yscale_text = text(length(data)+200, 0, '20 \muV','fontweight','b');

% Initialize NaN vectors for each channel
highlight_pz = nan(1,length(data));
highlight_oz = nan(1,length(data));
highlight_win = 6*fs : 9*fs;

% Add eyes open/closed labels using titles
ax_open = axes('color','w','position',[.4, ax(1).Position(2) + ax(1).Position(4), .2, .01]);
ax_open.Title.String    = 'Eyes Open';
ax_open.Title.FontSize  = 12;
ax_open.XColor = 'none';
ax_open.YColor = 'none';

ax_imopen = axes('color','w','position',[.335, ax(1).Position(2) + ax(1).Position(4) - ax(end).Position(2)/2, .1, .1]);
I_open    = imread(fullfile('misc','eye_open.png'));
I_closed  = imread(fullfile('misc','eye_closed.png'));
imshow(I_open);

%%
export_fig eeg_raster_1_V3.png -png -r300

%% Add topoplot to show alpha power
% Add topoplot
ax_topo  = axes('position',[0.675 0.7 0.275 0.275]);

% Get channel locations
[chanlocs, labels, Th, Rd, indices] = readlocs(fullfile(cd,'data','MotionTestChannelLocations_noEOG.ced'));

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%% TOPOPLOT HACK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This section is for updating the topoplot. This is a very hacky way of
% doing this, but for this example it will suffice. In order to update the
% topoplot in the loop, we either need to remake the topoplot every
% iteration (VERY BAD!) or go into the code and determine how the scalp map
% is being computed. That is what is happening below. Here, all the
% parameters required to compute the surface plot are being precomputed.
% Thus, only the surface plot will need to be updated each iteration and
% the new color data will be updated in the topoplot.

Th = pi/180*Th;
[x,y]     = pol2cart(Th,Rd);
plotrad = min(1.0,max(Rd)*1.02);            % default: just outside the outermost electrode location
plotrad = max(plotrad,0.5);                 % default: plot out to the 0.5 head boundary
% default_intrad = 1;     % indicator for (no) specified intrad
% intrad = min(1.0,max(Rd)*1.02);             % default: just outside the outermost electrode location
 rmax = 0.5;
% headrad = rmax;
% allx      = x;
% ally      = y;
squeezefac = rmax/plotrad;
intRd = Rd;
intTh = Th;
intx = x;
inty = y;
% intRd = intRd*squeezefac; % squeeze electrode arc_lengths towards the vertex
% Rd = Rd*squeezefac;       % squeeze electrode arc_lengths towards the vertex
%                           % to plot all inside the head cartoon
intx = intx*squeezefac;
inty = inty*squeezefac;
% x    = x*squeezefac;
% y    = y*squeezefac;
% allx    = allx*squeezefac;
% ally    = ally*squeezefac;
xi = linspace(-.5,0.5,67);   % x-axis description (row vector)
yi = linspace(-.5,0.5,67);
delta = xi(2)-xi(1); % length of grid entry

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Get alpha power limits
log_alpha = 10.*log10(all_alpha_power);
change_alpha = log_alpha - mean(log_alpha(:));
max_alpha = max(abs(change_alpha(:)));

% Compute alpha power in window
alpha_power_window = 10.*log10(sum(abs(all_alpha_data(:,round(midpoint-win_length/2):(round(midpoint-win_length/2)+win_length))).^2,2));
% Plot data using topoplot
tplot = topoplot(alpha_power_window- mean(log_alpha(:)),chanlocs,'electrodes','off','numcontour',4,'whitebk','on','headrad',0.5);
hold on;
% Add topoplot showing channels with time series
tplot2 = topoplot([],chanlocs(chans2plot),'style','blank','electrodes','labels','hcolor','none','whitebk','on');

% Change scale
cbar = colorbar;
cbar.Position(1)      = .925;
cbar.FontSize         = 9;
cbar.Label.String     = 'dB';
cbar.Label.Rotation   = 0;
cbar.Label.FontWeight = 'b';
%cbar.Limits = [-max_alpha max_alpha];
colorscale = flipud(lbmap(100,'BrownBlue'));%flipud(cbrewer('div','RdBu',100));%lbmap(100,'BrownBlue');
colormap(colorscale);

% % Turn off axes
% ax_topo.XColor = 'none';
% ax_topo.YColor = 'none';

% Change figure color back to white
fig3.Color = 'w';
%%
% export_fig eeg_raster_2_V3.png -png -r300

%% Animate
% Define values for animation
win         = [1 10*fs];
update_rate = 1/10;
win_length  = 1*fs;
count       = 1
filename = 'eeg_raster_finalanimated_v3_5.gif'

% Create play toggle button
PlayButton = uicontrol('Parent',fig3,'Style','togglebutton','String','Play','Units','normalized','Position',[0.01 0.01 .1 .05],'Visible','on',...
    'BackgroundColor','w');
% Create close button
CloseButton = uicontrol('Parent',fig3,'Style','togglebutton','String','Close','Units','normalized','Position',[0.89 0.01 .1 .05],'Visible','on',...
    'BackgroundColor','w');
% Pause to render
pause(0.01);

% Begin loop
start_time = tic;
while win(end) < size(alpha_data,2) 
    if PlayButton.Value == 1
        PlayButton.String = 'Pause';
        if toc(start_time) < update_rate
            % do nothing
        else
            for ii = 1: size(alpha_data,1)
                ax(ii).Children(end).YData = alpha_data(ii,win(1):win(2));
                ax(ii).Children(end-1).YData = alpha_highlight(ii,win(1):win(2));
            end
            % Uncomment if debugging eyes open/closed above:
            % peyes_open.YData = eyes_open(win(1):win(2));
            midpoint= round((win(1)+win(2))/2);
            % Compute alpha power in window
            alpha_power_window = 10.*log10(sum(abs(all_alpha_data(:,round(midpoint-win_length/2):(round(midpoint-win_length/2)+win_length))).^2,2));
            % Plot data using topoplot
            changeInAlpha =alpha_power_window- mean(log_alpha(:));
            % Compute updated grid
            [Xi,Yi,Zi] = griddata(inty,intx,double(changeInAlpha),yi',xi,'v4'); % interpolate data
            % Mask - EEGLAB stuff
            mask = (sqrt(Xi.^2 + Yi.^2) <= 0.5); % mask outside the plotting circle
            ii = find(mask == 0);
            Zi(ii)  = NaN;                         % mask non-plotting voxels with NaNs
            % UPDATE TOPOPLOT
            tplot.CData = Zi;
            % Get index of first point entering rectangle
            xIDX = round(midpoint + win_length/2);
            switch eyes_open(xIDX)
                % Change to eyes closed
                case 0
                    ax_open.Title.String     = 'Eyes Closed';
                    ax_imopen.Children.CData = I_closed;
                    % Change to eyes open
                case 1
                    ax_open.Title.String     = 'Eyes Open';
                    ax_imopen.Children.CData = I_open;
            end
            % Shift window
            win = win + fs*update_rate;
            % Pause to allow everything to render
            pause(0.001);
            % Update start time
            start_time = tic;
        end
    elseif PlayButton.Value == 0
        PlayButton.String = 'Play';
    end
    pause(0.001);
    % Check if close button pressed
    if CloseButton.Value == 1
       break; 
    end
    pause(0.001);
    
    % Capture the plot as an image
    frame = getframe(fig3);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    % Write to the GIF File
    if count == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',0.085);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',0.085);
    end
    count = count + 1;
end
close(fig3);

