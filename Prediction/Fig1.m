% Figure 1. Generating the predictions for the early and late noise
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
%% Early and late noise only
filename = sprintf('SNR_Ovlp_Choice');
V1mean = 88;
V2mean = 83;
V3 = linspace(0, V1mean, 50)';
V1 = V1mean*ones(size(V3));
V2 = V2mean*ones(size(V3));
nsmpls = 1024*1e3;
% simulation
matfile = fullfile(sim_dir, [filename, '.mat']);
if ~exist(matfile, 'file')
    eps = 9;
    eta = 0;
    sdV1 = eps*ones(size(V3));
    sdV2 = eps*ones(size(V3));
    sdV3 = eps*ones(size(V3));
    dat = table(V1,V2,V3,sdV1,sdV2,sdV3);
    pars = [eta, 1, 1, 1];
    reps = 40;
    tmp1a = nan([reps, 3, numel(V3)]);
    tmp2a = nan([reps, 3, numel(V3)]);
    tmp3a = nan([reps, 3, numel(V3)]);
    tmp1b = nan([reps, 3, numel(V3)]);
    tmp2b= nan([reps, 3, numel(V3)]);
    tmp3b = nan([reps, 3, numel(V3)]);
    parfor ri = 1:reps
        [tmp1a(ri,:,:), tmp2a(ri,:,:), tmp3a(ri,:,:)] = dDNaFig1(pars, dat, nsmpls);
        [tmp1b(ri,:,:), tmp2b(ri,:,:), tmp3b(ri,:,:)] = dDNbFig1(pars, dat, nsmpls);
    end
    probsa = squeeze(mean(tmp1a, 1));
    Ovlpsa = squeeze(mean(tmp2a, 1));
    CVsa = squeeze(mean(tmp3a, 1));
    probsb = squeeze(mean(tmp1b, 1));
    Ovlpsb = squeeze(mean(tmp1b, 1));
    CVsb = squeeze(mean(tmp3b, 1));

    eps = 0;
    eta = 4;
    sdV1 = eps*ones(size(V3));
    sdV2 = eps*ones(size(V3));
    sdV3 = eps*ones(size(V3));
    dat = table(V1,V2,V3,sdV1,sdV2,sdV3);
    pars = [eta, 1, 1, 1];
    tmp1a = nan([reps, 3, numel(V3)]);
    tmp2a = nan([reps, 3, numel(V3)]);
    tmp3a = nan([reps, 3, numel(V3)]);
    tmp1b = nan([reps, 3, numel(V3)]);
    tmp2b= nan([reps, 3, numel(V3)]);
    tmp3b = nan([reps, 3, numel(V3)]);
    parfor ri = 1:reps
        [tmp1a(ri,:,:), tmp2a(ri,:,:), tmp3a(ri,:,:)] = dDNaFig1(pars, dat, nsmpls);
        [tmp1b(ri,:,:), tmp2b(ri,:,:), tmp3b(ri,:,:)] = dDNbFig1(pars, dat, nsmpls);
    end
    probsaL = squeeze(mean(tmp1a, 1));
    OvlpsaL = squeeze(mean(tmp2a, 1));
    CVsaL = squeeze(mean(tmp3a, 1));
    probsbL = squeeze(mean(tmp1b, 1));
    OvlpsbL = squeeze(mean(tmp2b, 1));
    CVsbL = squeeze(mean(tmp3b, 1));
    save(matfile, "probsa",  "probsb", "Ovlpsa", "Ovlpsb", "CVsa", "CVsb",  "probsaL", "probsbL", "OvlpsaL", "OvlpsbL", "CVsaL", "CVsbL")
else
    load(matfile);
end
%%
h = figure;
subplot(3,2,1); hold on;
plot(V3, 1./CVsa(1,:), 'r:', 'LineWidth', 1);
plot(V3, 1./CVsb(1,:), 'r-', 'LineWidth', 1);
plot(V3, 1./CVsa(2,:), 'r:', 'LineWidth', 1);
plot(V3, 1./CVsb(2,:), 'r-', 'LineWidth', 1);
xlabel('V3');
ylabel('SNR');
xlim([0, V1mean]);
% yticks = get(gca, 'YTick');
% yticklabels = arrayfun(@(v) sprintf('%0.3f', v), yticks, 'UniformOutput', false);
% set(gca, 'YTickLabel', yticklabels);
legend({'Theoritical','Biological'}, 'Location', 'northeast', 'FontSize',10);
mysavefig(h, filename, plot_dir, 12, [6, 8]);

subplot(3,2,2); hold on;
plot(V3, 1./CVsaL(1,:), 'c:', 'LineWidth', 1);
plot(V3, 1./CVsbL(1,:), 'c-', 'LineWidth', 1);
plot(V3, 1./CVsaL(2,:), 'c:', 'LineWidth', 1);
plot(V3, 1./CVsbL(2,:), 'c-', 'LineWidth', 1);
xlabel('V3');
ylabel('SNR');
xlim([0, V1mean]);
% yticks = get(gca, 'YTick');
% yticklabels = arrayfun(@(v) sprintf('%0.3f', v), yticks, 'UniformOutput', false);
% set(gca, 'YTickLabel', yticklabels);
mysavefig(h, filename, plot_dir, 12, [6, 8]);

subplot(3,2,3); hold on;
plot(V3, Ovlpsa(1,:), 'r:', 'LineWidth', 1);
plot(V3, Ovlpsb(1,:), 'r-', 'LineWidth', 1);
xlabel('V3');
ylabel('% Overlap | V1, V2');
xlim([0, V1mean]);
% yticks = get(gca, 'YTick');
% yticklabels = arrayfun(@(v) sprintf('%2.1f', v), yticks, 'UniformOutput', false);
% set(gca, 'YTickLabel', yticklabels);
mysavefig(h, filename, plot_dir, 12, [6, 8]);

subplot(3,2,4); hold on;
plot(V3, OvlpsaL(1,:), 'c:', 'LineWidth', 1);
plot(V3, OvlpsbL(1,:), 'c-', 'LineWidth', 1);
xlabel('V3');
ylabel('% Overlap | V1, V2');
xlim([0, V1mean]);
% yticks = get(gca, 'YTick');
% yticklabels = arrayfun(@(v) sprintf('%2.1f', v), yticks, 'UniformOutput', false);
% set(gca, 'YTickLabel', yticklabels);
mysavefig(h, filename, plot_dir, 12, [6, 8]);

subplot(3,2,5); hold on;
ratio = probsa(1,:)./(probsa(1,:) + probsa(2,:))*100;
plot(V3, ratio, 'r:', 'LineWidth', 1);
plot([V1mean, V2mean], [1, 1]*min(ratio), 'kv', 'MarkerFaceColor','k');
ratio = probsb(1,:)./(probsb(1,:) + probsb(2,:))*100;
plot(V3, ratio, 'r-', 'LineWidth', 1);
xlabel('V3');
ylabel('% Correct | V1, V2');
xlim([0, V1mean]);
% yticks = get(gca, 'YTick');
% yticklabels = arrayfun(@(v) sprintf('%2.1f', v), yticks, 'UniformOutput', false);
% set(gca, 'YTickLabel', yticklabels);
mysavefig(h, filename, plot_dir, 12, [6, 8]);

subplot(3,2,6); hold on;
ratio = probsaL(1,:)./(probsaL(1,:) + probsaL(2,:))*100;
plot(V3, ratio, 'c:', 'LineWidth', 1);
plot([V1mean, V2mean], [1, 1]*min(ratio), 'kv', 'MarkerFaceColor', 'k');
ratio = probsbL(1,:)./(probsbL(1,:) + probsbL(2,:))*100;
plot(V3, ratio, 'c-', 'LineWidth', 1);
xlabel('V3');
ylabel('% Correct | V1, V2');
xlim([0, V1mean]);
ylim([61,65]);
% yticks = get(gca, 'YTick');
% yticklabels = arrayfun(@(v) sprintf('%2.1f', v), yticks, 'UniformOutput', false);
% set(gca, 'YTickLabel', yticklabels);
mysavefig(h, filename, plot_dir, 12, [6, 8]);