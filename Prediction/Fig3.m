% Figure 3. Predictions on the design matrix
%% define directories
[os, ~, ~] = computer;
if strcmp(os,'MACI64')
    rootdir = '/Users/bs3667/Dropbox (NYU Langone Health)/Bo Shen Working files/NoiseProject';
    Gitdir = '~/Noise';
elseif strcmp(os,'GLNXA64')
    rootdir = '/gpfs/data/glimcherlab/BoShen/Noise';
    Gitdir = '/gpfs/data/glimcherlab/BoShen/Noise';
end
plot_dir = fullfile(rootdir, 'Prediction');
sim_dir = fullfile(Gitdir, 'Prediction');
addpath(genpath(Gitdir));

%% graded color, two panels
V1mean = 83;% 53;
V2mean = 88;% 58;
epsV1 = 9; % early noise for V1
epsV2 = 9; % early noise for V2
V3 = linspace(0, V1mean, 25)';
V1 = V1mean*ones(size(V3));
V2 = V2mean*ones(size(V3));
sdV1 = epsV1*ones(size(V3))/2;
sdV2 = epsV2*ones(size(V3))/2;
chosenItem = randi(3, size(V3)); % dummy variable required by the function, meaningless used here
etavec = linspace(.8, 1.9, 8); % different levels of late noise
filename = sprintf('Ratio_Model_%iv3max%1.0f_%s', numel(V3), max(V3), '2Panels');
% simulation
SimDatafile = fullfile(sim_dir, [filename, '.mat']);
if exist(SimDatafile,'file')
    load(SimDatafile);
else
    Ratios = nan(2, numel(etavec), numel(V3));
    DNOverlaps = Ratios;
    SVOverlaps = Ratios;
    tmp = nan(2, numel(etavec), 3, numel(V3));
    DNPStats = tmp;
    V3Stats = tmp;
    SVStats = tmp;
    
    for i = 1:2
        if i == 1
            v = 0;
            eps = 4;
        elseif i == 2
            v = 1;
            eps = 9;
        end
        sdV3 = eps*ones(size(V3));
        dat = table(V1,V2,V3,sdV1,sdV2,sdV3, chosenItem);
        for ti = 1:numel(etavec)
            eta = etavec(ti);
            x = [1, 1, eta]; % M , w, eta
            [probs, dDNPQntls, V3Qntls, dSVQtls, dDNOvlp, dSVOvlp] = dDN(x, dat, 'absorb');
            DNPStats(i, ti, :, :) = dDNPQntls;
            DNOverlaps(i, ti, :) = dDNOvlp;
            V3Stats(i, ti, :, :) = V3Qntls;
            SVStats(i, ti, :, :) = dSVQtls;
            SVOverlaps(i, ti, :) = dSVOvlp;
            Ratios(i,ti,:) = probs(2,:)./(probs(1,:) + probs(2,:));
        end
    end
    xval = V3'/V1mean;
    save(SimDatafile, "Ratios","DNPStats","V3Stats","SVStats","DNOverlaps","SVOverlaps","xval",'-mat');
end
%% visualization
mycols = [0         0    1.0000
         0    0.3333    0.8333
         0    0.6667    0.6667
         0    1.0000    0.5000
    1.0000    0.7500         0
    1.0000    0.5000         0
    1.0000    0.2500         0
    1.0000         0         0];

xval = V3'/V1mean;
lt = 0.2;
rt = 0.8;
mask = xval >= lt & xval <= rt;
slope = [];
cmap = [];
h = figure;
filename = sprintf('Ratio_Model_%iv3max%1.0f_%s', numel(V3), max(V3), '2Panels');
for i = 1:2
    subplot(1, 2, 3-i); hold on;
    if i == 1
        v = 0;
        eps = 4;
        startColor = mycols(4,:); % Light-blue
        endColor = mycols(1,:);  % Blue
    elseif i == 2
        v = 1;
        eps = 9;
        startColor = mycols(8,:); % Red
        endColor = mycols(5,:); % Pink
    end
    cmap(:,:,i) = GradColor(startColor, endColor, numel(etavec));
    
    
    lg = [];
    for ti = 1:numel(etavec)
        ratio = squeeze(Ratios(i, ti, :))'*100;
        lg(ti) = plot(xval, ratio, '-', 'LineWidth', 2, 'Color', cmap(ti,:,i));
        coefficients = polyfit(xval(mask), ratio(mask), 1);
        slope(ti,i) = coefficients(1);
    end
    plot([V1mean, V2mean]/V1mean, [1, 1]*min(ratio(:)), 'kv', 'MarkerFaceColor', 'k', 'MarkerSize', 5);
    plot([lt, lt], [ylim], 'k--');
    plot([rt, rt], [ylim], 'k--');
    mylg = legend(lg, {'0.80','0.96','1.11','1.27','1.43','1.59','1.74','1.90'}, 'Location', "northeastoutside", 'Box','off');
    title(mylg, 'Late noise');
    xlabel('Scaled V3');
    ylabel('% Correct | V1, V2');
    mysavefig(h, filename, plot_dir, 12, [7, 2.8]);
end

%% slopes as bars
h = figure; hold on;
filename = sprintf('Ratio_Model_%iv3max%1.0f_%s', numel(V3), max(V3), 'Slopes');
mybar = bar(etavec, fliplr(slope), 1, 'FaceColor','flat');
for k = 1:size(slope,2)
    mybar(k).CData = cmap(:,:,k);
end
legend({'Ambiguous','Definitive'});
ylabel('Slope');
xlabel('\sigma_{Late}');
mysavefig(h, filename, plot_dir, 12, [2.7, 2.8]);

%% functions
function cmap = GradColor(startColor, endColor, numColors)
% Generate the colormap
cmap = zeros(numColors, 3); % Initialize the colormap matrix
for i = 1:3
    cmap(:, i) = linspace(startColor(i), endColor(i), numColors);
end
end