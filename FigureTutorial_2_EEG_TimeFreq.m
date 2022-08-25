%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                   EXAMPLE 2: TIME FREQUENCY ANALYSIS                   %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The following code is adapted from a tutorial provided by Mike X. Cohen
% on his website. The purpose of this tutorial is to highlight the effects
% of limits and scaling on visualization of time-freq data. We will use the
% dataset and wavelet method in the original tutorial. The tutorial can be
% found here: 
%
%     http://mikexcohen.com/lecturelets/whichbaseline/whichbaseline.html
%
% The data (provided by Mike) can be found here:
% 
% http://mikexcohen.com/book/AnalyzingNeuralTimeSeriesData_MatlabCode.zip
%
% His book, Analzyzing Neural Time Series Data, is an excellent resource for
% anyone working on EEG analysis:
%
% https://mitpress.mit.edu/books/analyzing-neural-time-series-data
%
% Code rewritten by: Justin Brantley
% email: justin dot a dot brantley at gmail dot com

clc
clear
close all;

% Simulated data - three frequencies
fs         = 100;        % sampling rate
move_freq  = [5 13 18];  % frequencies
Tau        = 10;         % time constant
time_vec   = 0:1/fs:Tau; % time vector
amplitude  = [-1;0;1];   % Amplitude of sinewave
burst_time = [2 8;       % Busrts of sinewave
              1 4;
              5 9];
          
% Generate matrix of Gaussian random noise:
swave = 0.01.*randn(3,length(time_vec));
% Add sine wave to noise matrix
for ii = 1:size(swave,1)
    xx   = burst_time(ii,1)*fs:burst_time(ii,2)*fs;
    s1    =  cos(move_freq(ii)*2*pi*time_vec(xx)+pi);
    swave(ii,xx) = swave(ii,xx) + s1;
end

% Plot sinewaves - the offsets are arbitrary and only for visualization
figure('color','w'); hold on; 
plot(swave(1,:) + 2.5); text(-1*fs,2.5,'5 Hz') % 5 hz
plot(swave(2,:) + 0  ); text(-1*fs, 0 ,'13 Hz') % 13 Hz
plot(swave(3,:) - 2.5); text(-1*fs,-2.5,'18 Hz') % 18 Hz
plot(sum(swave,1) - 6); text(-1*fs,-6,'Sum') % Sum of all three

% Clean up figure
xlim([0 Tau*fs])
ylim([-9 5]); 
ax = gca;
ax.XColor = 'none'; ax.YColor = 'none';
% % If you prefer a legend:
% leg = legend({'5 Hz','13 Hz','18 Hz','Sum'});
% leg.Location = 'northoutside';
% leg.Orientation = 'horizontal';
% leg.Box = 'off';

export_fig figuretutorial2_fig1.png -png -r300
%% Run spectogram
% Define spectrogram parameters
winSize = fs;
winOverlap = winSize - 1;
nfft = 1024;
minFreq = 1; % Min frequency for TF
maxFreq = 25; % Max frequency for TF

% Run spectrogram 
[tfmap,allfreq,time] = spectrogram(sum(swave,1),hann(winSize),winOverlap,nfft,fs,'reassign','yaxis');
freqIDX = find((allfreq<=maxFreq).*(allfreq>=minFreq));
tfmap   = tfmap(freqIDX,:);
freq    = allfreq(freqIDX);

% Plot time freq results
figure('color','w');

% Plot time series
ax1 = axes('position',[0.1 0.8 0.75 0.15]);
sumwave = sum(swave,1);
pp = plot(sumwave(0.5*fs:end-0.5*fs),'k'); xlim([0 length(pp.XData)]); 
ax1.XColor = 'none'; ax1.YColor = 'none';

% Plot time freq
ax2 = axes('position',[0.1 0.05 0.77 0.7]);
ss = surface(1:size(tfmap,2),freq,(abs(tfmap).^2)); ss.EdgeColor= 'none';
%contourf(1:size(tfmap,2),freq,(abs(tfmap).^2),40,'linecolor','none');

% Clean up
ax2.XTick = [];
ax2.YLabel.String = 'Frequency (Hz)';
ax2.XLabel.String = 'Time';
ax2.Box = 'on';
ax2.XLim = [1 size(tfmap,2)];
ax2.YLim = [minFreq+.1 maxFreq];

% Color bar 
cc = colorbar;
cc.Position(1) = .9;

export_fig figuretutorial2_fig2.png -png -r300

%% Change color - using divergent from before
colorscale = flipud(cbrewer('div','RdBu',100));
colorscale(colorscale < 0) = 0;
ax2.Colormap = colorscale;

export_fig figuretutorial2_fig3.png -png -r300

%% Change color - using sequential from before
colorscale = cbrewer('seq','Oranges',100);
colorscale(colorscale < 0) = 0;
ax2.Colormap = colorscale;

export_fig figuretutorial2_fig4.png -png -r300

%% Rescale data by subtracting mean across time
% This is known as an ERSP (event related spectral perturbation) in EEG
% analysis. Normally computed in log scale
ERSP = abs(tfmap).^2 - mean((abs(tfmap).^2),2);
ss.CData = ERSP;
% Update colorbar
minVal = min(ERSP(:));
maxVal = max(ERSP(:));
cc.Limits = [minVal maxVal];

%% Change back to divergent color scale
colorscale = flipud(cbrewer('div','RdBu',100));
colorscale(colorscale < 0) = 0;
ax2.Colormap = colorscale;

export_fig figuretutorial2_fig5.png -png -r300

%% What if the (+) and (-) values werent quite symmetric? 
ERSP2 = ERSP;
ERSP2(ERSP<0) = ERSP(ERSP<0)*0.5;
ss.CData = ERSP2;
% Update colorbar
minVal2 = min(ERSP2(:));
maxVal2 = max(ERSP2(:));
cc.Limits = [minVal2 maxVal2];

export_fig figuretutorial2_fig6.png -png -r300

%% Instead, we will compute the maximum of the min and max values
absMaxVal = max([abs(minVal2) abs(maxVal2)]);
ax2.CLim  = [-absMaxVal absMaxVal];
cc.Limits = [-absMaxVal absMaxVal];

%% Instead, we will compute the maximum of the min and max values
absMaxVal = max([abs(minVal) abs(maxVal)]);
ss.CData = ERSP;
ax2.CLim  = [-absMaxVal absMaxVal];
cc.Limits = [-absMaxVal absMaxVal];


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                      BEGIN MIKE COHEN CODE                             %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear
close all;

% Now, Lets look at some real EEG
load(fullfile('data','sampleEEGdata.mat'));

% specify baseline periods for dB-conversion
baseline_windows = [ -500 -200;
                     -100    0;
                       -1000 1000;% 0  300;
                     -800    0;
                   ];
               
% convert baseline time into indices
baseidx = reshape( dsearchn(EEG.times',baseline_windows(:)), [],2);

%% setup wavelet parameters

% frequency parameters
min_freq =  2;
max_freq = 30;
num_frex = 40;
frex = linspace(min_freq,max_freq,num_frex);

% which channel to plot
channel2use = 'o1';

% other wavelet parameters
range_cycles = [ 4 10 ];

s = logspace(log10(range_cycles(1)),log10(range_cycles(end)),num_frex) ./ (2*pi*frex);
wavtime = -2:1/EEG.srate:2;
half_wave = (length(wavtime)-1)/2;

% FFT parameters
nWave = length(wavtime);
nData = EEG.pnts * EEG.trials;
nConv = nWave + nData - 1;

% now compute the FFT of all trials concatenated
alldata = reshape( EEG.data(strcmpi(channel2use,{EEG.chanlocs.labels}),:,:) ,1,[]);
dataX   = fft( alldata ,nConv );

% initialize output time-frequency data
tf = zeros(size(baseidx,1),length(frex),EEG.pnts);

%% now perform convolution

% loop over frequencies
for fi=1:length(frex)
    
    % create wavelet and get its FFT
    wavelet  = exp(2*1i*pi*frex(fi).*wavtime) .* exp(-wavtime.^2./(2*s(fi)^2));
    waveletX = fft(wavelet,nConv);
    waveletX = waveletX ./ max(waveletX);
    
    % now run convolution in one step
    as = ifft(waveletX .* dataX);
    as = as(half_wave+1:end-half_wave);
    
    % and reshape back to time X trials
    as = reshape( as, EEG.pnts, EEG.trials );
    
    % compute power and average over trials
    tf(4,fi,:) = mean( abs(as).^2 ,2);
end

% db conversion and plot results

% define color limits
climdb  = [-3 3]; 
climpct = [-90 90];

% create new matrix for percent change
tfpct = zeros(size(tf));

for basei=1:size(tf,1)
    
    activity = tf(4,:,:);
    baseline = mean( tf(4,:,baseidx(basei,1):baseidx(basei,2)) ,3);
    
    % decibel
    tf(basei,:,:) = 10*log10( bsxfun(@rdivide, activity, baseline) );
    
    % percent change
    tfpct(basei,:,:) = 100 * bsxfun(@rdivide, bsxfun(@minus,activity,baseline), baseline);
end


%% plot results - no control for color limits
for basei=1:size(baseline_windows,1)
    
    % first plot dB
    figure(1), subplot(2,2,basei)
  
    contourf(EEG.times,frex,squeeze(tf(basei,:,:)),40,'linecolor','none')
    % JB EDIT:
    %set(gca,'clim',climdb,'ydir','normal','xlim',[-300 1000])
    set(gca,'ydir','normal','xlim',[-300 1000])
    cbar = colorbar;
    cbar.Label.String = 'dB';
    cbar.Label.Rotation = 0;
    cbar.Label.FontWeight = 'b';
    set(gcf,'color','w');
    % end JB EDIT
    
    title([ 'DB baseline of ' num2str(baseline_windows(basei,1)) ' to ' num2str(baseline_windows(basei,2)) ' ms' ])
    axtemp = gca;
    axtemp.Title.Position(2) = 31;
    
    % now plot percent change
    figure(2), subplot(2,2,basei)
    set(gcf,'color','w');
    contourf(EEG.times,frex,squeeze(tfpct(basei,:,:)),40,'linecolor','none')
    
    % JB EDIT:
    %set(gca,'clim',climpct,'ydir','normal','xlim',[-300 1000])
    set(gca,'ydir','normal','xlim',[-300 1000])
    cbar = colorbar;
    cbar.Label.String = '%';
    cbar.Label.Rotation = 0;
    cbar.Label.FontWeight = 'b';
    set(gcf,'color','w');
    % end JB EDIT
    
    title([ 'PCT baseline of ' num2str(baseline_windows(basei,1)) ' to ' num2str(baseline_windows(basei,2)) ' ms' ])
    axtemp = gca;
    axtemp.Title.Position(2) = 31;
    xlabel('Time (ms)'), ylabel('Frequency (Hz)')
end
%%
figure(1)
export_fig figuretutorial2_eegDB1.png -png -r300
figure(2)
export_fig figuretutorial2_eegpct1.png -png -r300

%% plot results - now control for color limits

for basei=1:size(baseline_windows,1)
    
    % first plot dB
    figure(3), subplot(2,2,basei)
   
    contourf(EEG.times,frex,squeeze(tf(basei,:,:)),40,'linecolor','none')
    set(gca,'clim',climdb,'ydir','normal','xlim',[-300 1000])
    % JB EDIT:
    cbar = colorbar;
    cbar.Label.String = 'dB';
    cbar.Label.Rotation = 0;
    cbar.Label.FontWeight = 'b';
    set(gcf,'color','w');
    % end JB EDIT
    
    title([ 'DB baseline of ' num2str(baseline_windows(basei,1)) ' to ' num2str(baseline_windows(basei,2)) ' ms' ])
    axtemp = gca;
    axtemp.Title.Position(2) = 31;
    xlabel('Time (ms)'), ylabel('Frequency (Hz)')
    
    % now plot percent change
    figure(4), subplot(2,2,basei)
    set(gcf,'color','w');
    contourf(EEG.times,frex,squeeze(tfpct(basei,:,:)),40,'linecolor','none')
    set(gca,'clim',climpct,'ydir','normal','xlim',[-300 1000])
    
    % JB EDIT:
    cbar = colorbar;
    cbar.Label.String = '%';
    cbar.Label.Rotation = 0;
    cbar.Label.FontWeight = 'b';
    set(gcf,'color','w');
    % end JB EDIT
    
    title([ 'PCT baseline of ' num2str(baseline_windows(basei,1)) ' to ' num2str(baseline_windows(basei,2)) ' ms' ])
    axtemp = gca;
    axtemp.Title.Position(2) = 31;
    xlabel('Time (ms)'), ylabel('Frequency (Hz)')
end

%%
figure(3)
export_fig figuretutorial2_eegDB2.png -png -r300

figure(4)
export_fig figuretutorial2_eegpct2.png -png -r300


%% plot results - change color scale

% JB EDIT:
colorscale = flipud(cbrewer('div','RdBu',100));
colorscale(colorscale < 0) = 0;

for basei=1:size(baseline_windows,1)
    
    % first plot dB
    figure(5), subplot(2,2,basei)
   
    contourf(EEG.times,frex,squeeze(tf(basei,:,:)),40,'linecolor','none')
    set(gca,'clim',climdb,'ydir','normal','xlim',[-300 1000])
    % JB EDIT:
    colormap(colorscale);
    cbar = colorbar;
    cbar.Label.String = 'dB';
    cbar.Label.Rotation = 0;
    cbar.Label.FontWeight = 'b';
    set(gcf,'color','w');
    % end JB EDIT
    
    title([ 'DB baseline of ' num2str(baseline_windows(basei,1)) ' to ' num2str(baseline_windows(basei,2)) ' ms' ])
    axtemp = gca;
    axtemp.Title.Position(2) = 31
    xlabel('Time (ms)'), ylabel('Frequency (Hz)')
    
    % now plot percent change
    figure(6), subplot(2,2,basei)
    set(gcf,'color','w');
    contourf(EEG.times,frex,squeeze(tfpct(basei,:,:)),40,'linecolor','none')
    set(gca,'clim',climpct,'ydir','normal','xlim',[-300 1000])
    
    % JB EDIT:
    colormap(colorscale);
    cbar = colorbar;
    cbar.Label.String = '%';
    cbar.Label.Rotation = 0;
    cbar.Label.FontWeight = 'b';
    set(gcf,'color','w');
    % end JB EDIT
    
    title([ 'PCT baseline of ' num2str(baseline_windows(basei,1)) ' to ' num2str(baseline_windows(basei,2)) ' ms' ])
    axtemp = gca;
    axtemp.Title.Position(2) = 31
    xlabel('Time (ms)'), ylabel('Frequency (Hz)')
end

%%
figure(5)
export_fig figuretutorial2_eegDB3.png -png -r300
figure(6)
export_fig figuretutorial2_eegpct3.png -png -r300
%% Finalize plot - JB edited

fig = figure('color','w');
ax = tight_subplot(2,2,[.05 .05],[.1 .1],[.1 .15]);
colorscale = flipud(cbrewer('div','RdBu',100));
colorscale(colorscale<0) = 0

for basei=1:size(baseline_windows,1)
    
    % first plot dB
    axes(ax(basei));
    box on;
    contourf(EEG.times,frex,squeeze(tf(basei,:,:)),40,'linecolor','none')
    set(gca,'clim',climdb,'ydir','normal','xlim',[-300 1000],'fontsize',9)
    
    % Add line to show show stimulus
    line([0 0],[min_freq max_freq],'color',0.5.*ones(3,1),'linestyle','--')
    
    % Add title to show baseline norm period
    title([ num2str(baseline_windows(basei,1)) ' to ' num2str(baseline_windows(basei,2)) ' ms' ])
    
    if basei < size(baseline_windows,1)-1
       ax(basei).XTickLabel = [];
    else
       ax(basei).XLabel.String = 'Time (ms)';
       ax(basei).XLabel.FontWeight = 'b';
    end
       
end

% Add title
ax_title = axes('position',[.4 .9 .2 .05]);
ax_title.Title.String = 'This Figure Explains Baseline Normalization in dB';
ax_title.Title.FontWeight = 'b';
ax_title.Title.FontSize = 12;
ax_title.Color = 'none';
ax_title.XColor = 'none';
ax_title.YColor = 'none';

% Change labels
ax(1).YLabel.String = 'Frequency (Hz)';
ax(1).YLabel.FontWeight = 'b';
ax(3).YLabel.String = 'Frequency (Hz)';
ax(3).YLabel.FontWeight = 'b';
ax(2).YTickLabels = [];
ax(4).YTickLabels = [];

% Add colorbar
ax_cbar = axes('position',[.85 .1 .01 .8]);
ax_cbar.FontSize = 9;
ax_cbar.XColor = 'none';
ax_cbar.YColor = 'none';
colormap(colorscale);
cbar = colorbar;
cbar.Limits = climdb;
ax_cbar.CLim = climdb;
cbar.Position(3) = .025;
cbar.Label.String = '(dB)';
cbar.Label.Rotation = 0;
cbar.Label.FontWeight = 'b';
cbar.Label.FontSize = 9;
cbar.Label.Position(1) = 3.5;
set(gcf,'color','w');

export_fig figuretutorial2_eegDBfinal.png -png -r300
