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
sim_dir = fullfile(Gitdir, 'Prediction');
addpath(genpath(Gitdir));

Npar = 40;
mypool = parpool(Npar);
%% Mixed noise 
V1mean = 88;
V2mean = 83;
V3 = linspace(0, V1mean, 50)';
V1 = V1mean*ones(size(V3));
V2 = V2mean*ones(size(V3));
nsmpls = 1024*1e3;

% h = figure; hold on;
% epsvec = linspace(0, 9, 8);
% etavec = linspace(3.63, 0, 8);
% plot(epsvec, etavec, '.-');
% phi = atan((epsvec/9)./(etavec/3.63));
%phi = linspace(0, pi/2, 8);
phi = [0, .2, .35, .47, .6, .7, .8, 1]*pi/2;
epsvec = sin(phi)*9;
etavec = cos(phi)*3.63;
% for i = 1:numel(epsvec)
%     plot([0,epsvec(i)], [0, etavec(i)],'b-');
% end
% phi = linspace(0, pi/2, 8);
% epsvec = sin(phi)*9;
% etavec = cos(phi)*3.63;
% for i = 1:numel(epsvec)
%     plot([0,epsvec(i)], [0, etavec(i)],'r-');
% end

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
        parfor ri = 1:40
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
% mycols = [winter(4); flip(autumn(5))];
% mycols(5,:) = [];
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
plot([V1mean, V2mean]/V2mean, [1, 1]*min(ratio(:)), 'kv', 'MarkerFaceColor', 'k');
xlim([0, V1mean/V2mean]);
plot([lt, lt], [max(ratio(:)), min(ratio(:))], 'k--');
plot([rt, rt], [max(ratio(:)), min(ratio(:))], 'k--');
mylg = legend(fliplr(lg), {'9.0 0.0',' ',' ',' ',' ',' ',' ','0.0 3.6'}, 'Location', "northeastoutside", 'Box','off');
title(mylg, 'Noise \sigma_{Early} \sigma_{Late}');
xlabel('Scaled V3');
ylabel('% Correct | V1, V2');
mysavefig(h, filename, plot_dir, 12, [5.4, 4]);
%%
h = figure; hold on;
filename = [filename, 'Slope'];
for i = length(slope):-1:1
    bar(9-i, slope(i), 'FaceColor',mycols(i,:));
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
epsvec = linspace(0, 13, 101);
%[epsmesh, V3mesh] = meshgrid(epsvec, V3vec);
%V3 = V3mesh(:);
%eps = epsmesh(:);

V1 = V1mean*ones(size(V3));
V2 = V2mean*ones(size(V3));
nsmpls = 1024*1e3;
etavec = linspace(0, 3.63*13/9, 100);

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
            reps = 40;
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

