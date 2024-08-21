% Figure 4. Predictions on the design matrix
%% define directories
DefineIO;
plot_dir = fullfile(rootdir, 'Prediction','Fig4');
sim_dir = fullfile(rootdir, 'Prediction','Fig4');
if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end
if ~exist(sim_dir,'dir')
    mkdir(sim_dir);
end

%% loading parrellel CPU cores
Myclust = parcluster();
Npar = Myclust.NumWorkers;
mypool = parpool(Npar/2);
reps = Npar; % 40; % repetition of simulations to make the results smooth

%% graded color, two panels
V1mean = 88;
V2mean = 83;
eps1 = 4; % early noise for V1
eps2 = 4; % early noise for V2
V3 = linspace(0, V2mean, 50)';
V1 = V1mean*ones(size(V3));
V2 = V2mean*ones(size(V3));
sdV1 = eps1*ones(size(V3));
sdV2 = eps2*ones(size(V3));
etavec = linspace(.8, 1.9, 8); % different levels of late noise
products = {'Probability'};
filename = sprintf('Ratio_Model_%iv3max%1.0f_%s', numel(V3), max(V3), '2Panels');
% simulation
SimDatafile = fullfile(sim_dir, [filename, '.mat']);
if exist(SimDatafile,'file')
    load(SimDatafile);
else
    Ratios = nan(2, numel(etavec), numel(V3));
    for i = 1:2
        if i == 1
            v = 0;
            eps = 4;
        elseif i == 2
            v = 1;
            eps = 9;
        end
        sdV3 = eps*ones(size(V3));
        dat = table(V1,V2,V3,sdV1,sdV2,sdV3);
        for ti = 1:numel(etavec)
            eta = etavec(ti);
            pars = [eta, 1, 1, 1];
            tmpb = nan([reps, numel(V3), 3]);
            parfor ri = 1:reps
                [tmpb(ri,:,:), ~, ~] = dnDNM(dat, pars, 'biological', products); % biological model
            end
            probs = squeeze(mean(tmpb, 1));
            Ratios(i,ti,:) = probs(:,1)./(probs(:,1) + probs(:,2))*100;
        end
    end
    xval = V3'/V2mean;
    save(SimDatafile, "Ratios","xval",'-mat');
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
etavec = etavec(1:6);
xval = V3'/V2mean;
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
        yrng = [68.3, 74.5]+1;
    elseif i == 2
        v = 1;
        eps = 9;
        startColor = mycols(8,:); % Red
        endColor = mycols(5,:); % Pink
        yrng = [67.1, 72.45]+1;
    end
    cmap(:,:,i) = GradColor(startColor, endColor, numel(etavec));
    
    
    lg = [];
    for ti = 1:numel(etavec)
        ratio = squeeze(Ratios(i, ti, :))';
        if ti == 2 || ti == 5
            lg(ti) = plot(xval, ratio, '-', 'LineWidth', 2, 'Color', cmap(ti,:,i));
        else
            lg(ti) = plot(xval, ratio, '--', 'LineWidth', 2, 'Color', cmap(ti,:,i));
        end
        coefficients = polyfit(xval(mask), ratio(mask), 1);
        slope(ti,i) = coefficients(1);
    end
    plot([V1mean, V2mean]/V2mean, [1, 1]*min(ratio(:)), 'kv', 'MarkerFaceColor', [.7,.7,.7], 'MarkerSize', 5);
    ylim(yrng);
    plot([lt, lt], [ylim], 'k--');
    plot([rt, rt], [ylim], 'k--');
    mylg = legend(lg, {'0.80','0.96','1.11','1.27','1.43','1.59'}, 'Location', "northeastoutside", 'Box','off');
    title(mylg, 'Late noise');
    xlabel('Scaled V3');
    ylabel('% Correct | V1, V2');
    mysavefig(h, filename, plot_dir, 12, [7.27, 2.37]);
end

%% slopes as bars
h = figure; hold on;
filename = sprintf('Ratio_Model_%iv3max%1.0f_%s', numel(V3), max(V3), 'Slopes');
mybar = barh(etavec', fliplr(slope), 1, 'FaceColor','flat');
for k = 1:size(slope,2)
    mybar(k).CData = cmap(:,:,3-k);
end
set(gca, 'YDir', 'reverse');
set(gca, 'XAxisLocation', 'top');
%set(gca, 'YAxisLocation', 'right');
set(gca, 'YColor', 'none')
set(gca, 'YTick', []);
%legend({'Vague','Precise'}, 'Location', 'northwest');
xlabel('Slope');
%ylabel('\sigma_{Late}');
mysavefig(h, filename, plot_dir, 12, [2.06, 1.59]*1.1);

%% V3 mean value - variance matrix
V1mean = 88;
V2mean = 83;
eps1 = 4; % early noise for V1
eps2 = 4; % early noise for V2
V3 = linspace(0, V1mean, 100)';
eps3 = linspace(0, 60, 101);
V1 = V1mean*ones(size(V3));
V2 = V2mean*ones(size(V3));
sdV1 = eps1*ones(size(V3));
sdV2 = eps2*ones(size(V3));
etavec = [1.0, 1.4286]; % 1, 1.4286; %linspace(.8, 1.9, 8); % different levels of late noise
products = {'Probability'};
for ti = 1:numel(etavec)
    eta = etavec(ti);
    filename = sprintf('Ratio_Model_%iv3max%1.0f_%ivar3max%1.0f_eta%1.2f', numel(V3), max(V3), numel(eps3), max(eps3), eta);
    % simulation
    SimDatafile = fullfile(sim_dir, [filename, '.mat']);
    if ~exist(SimDatafile,'file')
        Ratios = nan(numel(eps3), numel(V3));
        for i = 1:numel(eps3)
            fprintf('Mixed noise - late noise %i/%i, early noise %i/%i\n', ti, numel(etavec), i, numel(eps3));
            sdV3 = eps3(i)*ones(size(V3));
            dat = table(V1,V2,V3,sdV1,sdV2,sdV3);
            pars = [eta, 1, 1, 1];
            tmpb = nan([reps, numel(V3), 3]);
            parfor ri = 1:reps
                [tmpb(ri,:,:), ~, ~] = dnDNM(dat, pars, 'biological', products); % biological model
            end
            probs = squeeze(mean(tmpb, 1));
            Ratios(i,:) = probs(:,1)./(probs(:,1) + probs(:,2))*100;
        end
        save(SimDatafile, "Ratios", "V3", "eta", "V2mean", '-mat');
    end
end
%% visualization
filename = 'CardinalView_Model';
etavec = [1.0, 1.4286];
h = figure;
for ti = 1:2
    eta = etavec(ti);
    simrslts = sprintf('Ratio_Model_%iv3max%1.0f_%ivar3max%1.0f_eta%1.2f', numel(V3), max(V3), numel(eps3), max(eps3), eta);
    SimDatafile = fullfile(sim_dir, [simrslts, '.mat']);
    load(SimDatafile);

    subplot(2,2,1+(ti-1)*2); hold on;
    colormap("bone");
    cmap = bone(numel(eps3));
    for ri = 1:5:numel(eps3)
        plot(V3/V2mean, Ratios(ri,:), '.-', 'Color', cmap(ri,:));
    end
    xlabel('Scaled V3');
    ylabel('% Correct | V1, V2');
    title(sprintf('Late noise eta = %1.2f', eta));
    mysavefig(h, filename, plot_dir, 12, [10, 8]);

    subplot(2,2,2+(ti-1)*2);
    colormap("jet");
    imagesc(V3/V2mean, eps3/V2mean, Ratios, [65, 80]);
    title('');
    colorbar;
    xlabel('Scaled V3');
    ylabel('\sigma_3 (Early noise)');
    set(gca, 'YDir', 'normal');
    mysavefig(h, filename, plot_dir, 12, [10, 8]);
end
%% functions
function cmap = GradColor(startColor, endColor, numColors)
% Generate the colormap
cmap = zeros(numColors, 3); % Initialize the colormap matrix
for i = 1:3
    cmap(:, i) = linspace(startColor(i), endColor(i), numColors);
end
end