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

%% Mixed noise 
filename = sprintf('Choice_MixedNoise');
V1mean = 88;
V2mean = 83;
V3 = linspace(0, V1mean, 50)';
V1 = V1mean*ones(size(V3));
V2 = V2mean*ones(size(V3));
nsmpls = 1e7;

epsvec = linspace(0, 9, 8)/3;
etavec = linspace(3.63, 0, 8)/3;

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
        probsa(i,:,:) = dDNaFig2(pars, dat, nsmpls);
        probsb(i,:,:) = dDNbFig2(pars, dat, nsmpls);
    end
    save(matfile, "probsa",  "probsb");
else
    load(matfile);
end

%% visualization
h = figure; hold on;
gcolors = jet(9); %flip([winter(4); flip(autumn(4))]);
mycols =[0.8472126105344099, 0.2612072279892349, 0.30519031141868513;
    0.9637831603229527, 0.47743175701653207, 0.28581314878892733;
    0.9934640522875817, 0.7477124183006535, 0.4352941176470587;
    0.9977700884275279, 0.930872741253364, 0.6330642060745867;
    0.944252210688197, 0.9777008842752788, 0.6620530565167244;
    0.7477124183006538, 0.8980392156862746, 0.6274509803921569;
    0.4530565167243369, 0.7815455594002307, 0.6462898885044214;
    0.21607074202229912, 0.5556324490580546, 0.7319492502883507];
ratio = [];
slope = [];
lt = 0.2;
rt = 0.8;
for i = 1:numel(epsvec)
    x = V3/V2mean;
    mask = x >= lt & x <= rt;
    ratio(:,i,1) = squeeze(probsa(i,1,:)./(probsa(i,1,:) + probsa(i,2,:))*100);
    plot(x, ratio(:,i,1), ':', 'Color', mycols(i,:), 'LineWidth', 1);
    ratio(:,i,2) = squeeze(probsb(i,1,:)./(probsb(i,1,:) + probsb(i,2,:))*100);
    plot(x, ratio(:,i,2), '-', 'Color', mycols(i,:), 'LineWidth', 1.6);
    coefficients = polyfit(x(mask), ratio(mask,i,2), 1);
    slope(i) = coefficients(1);
end
plot([V1mean, V2mean]/V1mean, [1, 1]*min(ratio(:)), 'kv', 'MarkerFaceColor', 'k');
xlim([0, V1mean/V2mean]);
plot([lt, lt], [max(ratio(:)), min(ratio(:))], 'k--');
plot([rt, rt], [max(ratio(:)), min(ratio(:))], 'k--');
xlabel('Scaled V3');
ylabel('% Correct | V1, V2');
mysavefig(h, filename, plot_dir, 12, [5, 4]);
%%
h = figure; hold on;
filename = 'Fig2a_Inset';
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
V3vec = linspace(0, V1mean, 50);
x = V3vec/V2mean;
mask = x > .2 & x < .8;
epsvec = linspace(0, 13, 201);
[epsmesh, V3mesh] = meshgrid(epsvec, V3vec);
V3 = V3mesh(:);
eps = epsmesh(:);

V1 = V1mean*ones(size(V3));
V2 = V2mean*ones(size(V3));
nsmpls = 1e7;
etavec = linspace(0, 3.63*13/9, 201);

matfile = fullfile(sim_dir, [filename, '.mat']);
if ~exist(matfile, 'file')
    slope = nan([numel(epsvec), numel(etavec)]);
    for j = 1:numel(etavec)
        fprintf("Eta #%i\n",j);
        eta = etavec(j);
        sdV1 = eps;
        sdV2 = eps;
        sdV3 = eps;
        dat = table(V1,V2,V3,sdV1,sdV2,sdV3);
        pars = [eta, 1, 1, 1];
        tmp = dDNbFig2(pars, dat, nsmpls);
        ratio = tmp(1,:)./(tmp(1,:) + tmp(2,:))*100;
        ratio = reshape(ratio, numel(V3vec), numel(epsvec));
        for i = 1:numel(epsvec)
            coefficients = polyfit(x(mask), ratio(mask,i), 1);
            slope(i,j) = coefficients(1);
        end
    end
    save(matfile, 'slope');
else
    load(matfile);
end
%%
addpath(genpath('/Users/bs3667/Library/CloudStorage/GoogleDrive-bs3667@nyu.edu/My Drive/Mylib'));
h = figure;
hold on;
imagesc(etavec, epsvec, slope);
colormap(bluewhitered);
colorbar;
ylabel('\sigma_{Early noise}');
xlabel('\sigma_{Late noise}');
xlim([min(etavec), max(etavec)]);
ylim([min(epsvec), max(epsvec)]);
mysavefig(h, filename, plot_dir, 12, [5, 4]);

