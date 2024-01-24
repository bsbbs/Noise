% Figure 2. Intertwisted magnitude of noise 
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
sim_dir = fullfile(rootdir, 'Prediction');
addpath(genpath(Gitdir));

%% Mixed noise 

V1mean = 88;
V2mean = 83;
V3 = linspace(0, V1mean, 50)';
V1 = V1mean*ones(size(V3));
V2 = V2mean*ones(size(V3));
nsmpls = 1024*1e3;

epsvec = linspace(0, 9, 8)/2;
etavec = linspace(3.63, 0, 8)/2;
filename = sprintf('Choice_MixedNoise_eps%1.2f_eta%1.2f', max(epsvec), max(etavec));
% simulation
matfile = fullfile(sim_dir, [filename, '.mat']);
if ~exist(matfile, 'file')
    probsa = nan([numel(epsvec), 3, numel(V3)]);
    probsb = nan([numel(epsvec), 3, numel(V3)]);
    for i = 1:numel(epsvec)
        fprintf("Level #%i\n",i);
        eps = epsvec(i);
        eta = etavec(i);
        sdV1 = eps*ones(size(V3));
        sdV2 = eps*ones(size(V3));
        sdV3 = eps*ones(size(V3));
        dat = table(V1,V2,V3,sdV1,sdV2,sdV3);
        pars = [eta, 1, 1, 1];
        tmpa = nan([10, 3, numel(V3)]);
        tmpb = nan([10, 3, numel(V3)]);
        parfor ri = 1:120
            tmpa(ri,:,:) = dDNaFig2(pars, dat, nsmpls);
            tmpb(ri,:,:) = dDNbFig2(pars, dat, nsmpls);
        end
        probsa(i,:,:) = squeeze(mean(tmpa, 1));
        probsb(i,:,:) = squeeze(mean(tmpb, 1));
    end
    save(matfile, "probsa",  "probsb");
else
    load(matfile);
end

%% visualization
h = figure; hold on;
mycols =flip([0.8472126105344099, 0.2612072279892349, 0.30519031141868513;
    0.9637831603229527, 0.47743175701653207, 0.28581314878892733;
    0.9934640522875817, 0.7477124183006535, 0.4352941176470587;
    0.9977700884275279, 0.930872741253364, 0.6330642060745867;
    0.944252210688197, 0.9777008842752788, 0.6620530565167244;
    0.7477124183006538, 0.8980392156862746, 0.6274509803921569;
    0.4530565167243369, 0.7815455594002307, 0.6462898885044214;
    0.21607074202229912, 0.5556324490580546, 0.7319492502883507]);
mycols = jet(8);
mycols = [winter(4); flip(autumn(5))];
mycols(5,:) = [];
mycols = [0         0    1.0000
         0    0.3333    0.8333
         0    0.6667    0.6667
         0    1.0000    0.5000
    1.0000    0.7500         0
    1.0000    0.5000         0
    1.0000    0.2500         0
    1.0000         0         0];
ratio = [];
slope = [];
lt = 0.2;
rt = 0.8;
lg = [];
for i = 1:numel(epsvec)
    x = V3/V2mean;
    mask = x >= lt & x <= rt;
    ratio(:,i,1) = squeeze(probsa(i,1,:)./(probsa(i,1,:) + probsa(i,2,:))*100);
    plot(x, ratio(:,i,1), ':', 'Color', mycols(i,:), 'LineWidth', 2);
    ratio(:,i,2) = squeeze(probsb(i,1,:)./(probsb(i,1,:) + probsb(i,2,:))*100);
    lg(i) = plot(x, ratio(:,i,2), '-', 'Color', mycols(i,:), 'LineWidth', 2);
    coefficients = polyfit(x(mask), ratio(mask,i,2), 1);
    slope(i) = coefficients(1);
end
plot([V1mean, V2mean]/V1mean, [1, 1]*min(ratio(:)), 'kv', 'MarkerFaceColor', 'k');
xlim([0, V1mean/V2mean]);
plot([lt, lt], [max(ratio(:)), min(ratio(:))], 'k--');
plot([rt, rt], [max(ratio(:)), min(ratio(:))], 'k--');
mylg = legend(lg, {'0.0 1.2','0.4 1.0','0.9 0.9','1.3 0.7','1.7 0.5','2.1 0.3','2.6 0.2','3.0 0.0'}, 'Location', "northeastoutside", 'Box','off');
title(mylg, 'Noise \sigma_{Early} \sigma_{Late}');
xlabel('Scaled V3');
ylabel('% Correct | V1, V2');
mysavefig(h, filename, plot_dir, 12, [5.4, 4]);
%%
h = figure; hold on;
filename = [filename, 'Slope'];
for i = 1:length(slope)
    bar(i, slope(i), 'FaceColor',mycols(i,:));
end
ylabel('Slope');
xlabel('Cases');
mysavefig(h, filename, plot_dir, 12, [2, 2]);

%% Full range panel
filename = sprintf('Choice_MixedNoiseFullrng');
V1mean = 88;
V2mean = 83;
V3 = linspace(0, V1mean, 50)';
x = V3/V2mean;
mask = x > .2 & x < .8;
epsvec = linspace(0, 13, 201);
%[epsmesh, V3mesh] = meshgrid(epsvec, V3vec);
%V3 = V3mesh(:);
%eps = epsmesh(:);

V1 = V1mean*ones(size(V3));
V2 = V2mean*ones(size(V3));
nsmpls = 1024*1e3;
etavec = linspace(0, 3.63*13/9, 201);

matfile = fullfile(sim_dir, [filename, '.mat']);
if ~exist(matfile, 'file')
    slope = nan([numel(epsvec), numel(etavec)]);
    for j = 1:numel(etavec)
        fprintf("Eta #%i, Loop over Eps: ",j);
        eta = etavec(j);
        for i = 1:numel(epsvec)
            fprintf(".");
            eps = epsvec(i);
            sdV1 = eps*ones(size(V3));
            sdV2 = eps*ones(size(V3));
            sdV3 = eps*ones(size(V3));
            dat = table(V1,V2,V3,sdV1,sdV2,sdV3);
            pars = [eta, 1, 1, 1];
            reps = 120;
            tmp = nan([reps, 3, numel(V3)]);
            parfor ri = 1:reps
                tmp(ri,:,:) = dDNbFig2(pars, dat, nsmpls);
            end
            probs = squeeze(mean(tmp, 1));
            ratio = probs(1,:)./(probs(1,:) + probs(2,:))*100;
            %ratio = reshape(ratio, numel(V3), numel(epsvec));
            coefficients = polyfit(x(mask), ratio(mask), 1);
            slope(i,j) = coefficients(1);
        end
        fprintf('\n');
    end
    save(matfile, 'slope');
else
    load(matfile);
end
%%
% addpath(genpath('/Users/bs3667/Library/CloudStorage/GoogleDrive-bs3667@nyu.edu/My Drive/Mylib'));
h = figure;
hold on;
imagesc(etavec, epsvec, slope);
% colormap(bluewhitered);
colorbar;
ylabel('\sigma_{Early noise}');
xlabel('\sigma_{Late noise}');
xlim([min(etavec), max(etavec)]);
ylim([min(epsvec), max(epsvec)]);
mysavefig(h, filename, plot_dir, 12, [5, 4]);

