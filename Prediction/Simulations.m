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

%% Early and late noise only
filename = sprintf('CV_Ovlp_Choice');
V1mean = 88;
V2mean = 83;
V3 = linspace(0, V1mean, 40)';
V1 = V1mean*ones(size(V3));
V2 = V2mean*ones(size(V3));
nsmpls = 1e7;
% simulation
matfile = fullfile(plot_dir, [filename, '.mat']);
if ~exist(matfile, 'file')
    eps = 9;
    eta = 0;
    sdV1 = eps*ones(size(V3));
    sdV2 = eps*ones(size(V3));
    sdV3 = eps*ones(size(V3));
    dat = table(V1,V2,V3,sdV1,sdV2,sdV3);
    pars = [eta, 1, 1, 1];
    [probsa, Ovlpsa, CVsa] = dDNaFig1(pars, dat, nsmpls);
    [probsb, Ovlpsb, CVsb] = dDNbFig1(pars, dat, nsmpls);

    eps = 0;
    eta = 4;
    sdV1 = eps*ones(size(V3));
    sdV2 = eps*ones(size(V3));
    sdV3 = eps*ones(size(V3));
    dat = table(V1,V2,V3,sdV1,sdV2,sdV3);
    pars = [eta, 1, 1, 1];
    [probsaL, OvlpsaL, CVsaL] = dDNaFig1(pars, dat, nsmpls);
    [probsbL, OvlpsbL, CVsbL] = dDNbFig1(pars, dat, nsmpls);
    save(matfile, "probsa",  "probsb", "Ovlpsa", "Ovlpsb", "CVsa", "CVsb",  "probsaL", "probsbL", "OvlpsaL", "OvlpsbL", "CVsaL", "CVsbL")
else
    load(matfile);
end
%
h = figure;
subplot(3,2,1); hold on;
plot(V3, CVsa(1,:), 'k:', 'LineWidth', 1);
plot(V3, CVsb(1,:), 'k-', 'LineWidth', 1);
xlabel('V3');
ylabel('Coeff. of Var.');
xlim([0, V1mean]);
yticks = get(gca, 'YTick');
yticklabels = arrayfun(@(v) sprintf('%0.2f', v), yticks, 'UniformOutput', false);
set(gca, 'YTickLabel', yticklabels);
legend({'Theoritical','Biological'}, 'Location', 'northeast', 'FontSize',10);
mysavefig(h, sprintf('CV_Ovlp_Choice'), plot_dir, 12, [6, 8]);

subplot(3,2,2); hold on;
plot(V3, CVsaL(1,:), 'k:', 'LineWidth', 1);
plot(V3, CVsbL(1,:), 'k-', 'LineWidth', 1);
xlabel('V3');
ylabel('Coeff. of Var.');
xlim([0, V1mean]);
yticks = get(gca, 'YTick');
yticklabels = arrayfun(@(v) sprintf('%0.2f', v), yticks, 'UniformOutput', false);
set(gca, 'YTickLabel', yticklabels);
mysavefig(h, sprintf('CV_Ovlp_Choice'), plot_dir, 12, [6, 8]);

subplot(3,2,3); hold on;
plot(V3, Ovlpsa(1,:), 'k:', 'LineWidth', 1);
plot(V3, Ovlpsb(1,:), 'k-', 'LineWidth', 1);
xlabel('V3');
ylabel('% Overlap (V1 & V2)');
xlim([0, V1mean]);
yticks = get(gca, 'YTick');
yticklabels = arrayfun(@(v) sprintf('%0.1f', v), yticks, 'UniformOutput', false);
set(gca, 'YTickLabel', yticklabels);
mysavefig(h, sprintf('CV_Ovlp_Choice'), plot_dir, 12, [6, 8]);

subplot(3,2,4); hold on;
plot(V3, OvlpsaL(1,:), 'k:', 'LineWidth', 1);
plot(V3, OvlpsbL(1,:), 'k-', 'LineWidth', 1);
xlabel('V3');
ylabel('% Overlap (V1 & V2)');
xlim([0, V1mean]);
yticks = get(gca, 'YTick');
yticklabels = arrayfun(@(v) sprintf('%0.1f', v), yticks, 'UniformOutput', false);
set(gca, 'YTickLabel', yticklabels);
mysavefig(h, sprintf('CV_Ovlp_Choice'), plot_dir, 12, [6, 8]);

subplot(3,2,5); hold on;
ratio = probsa(1,:)./(probsa(1,:) + probsa(2,:))*100;
plot(V3, ratio, 'k:', 'LineWidth', 1);
ratio = probsb(1,:)./(probsb(1,:) + probsb(2,:))*100;
plot(V3, ratio, 'k-', 'LineWidth', 1);
xlabel('V3');
ylabel('% Correct (V1 & V2)');
xlim([0, V1mean]);
yticks = get(gca, 'YTick');
yticklabels = arrayfun(@(v) sprintf('%0.1f', v), yticks, 'UniformOutput', false);
set(gca, 'YTickLabel', yticklabels);
mysavefig(h, sprintf('CV_Ovlp_Choice'), plot_dir, 12, [6, 8]);

subplot(3,2,6); hold on;
ratio = probsaL(1,:)./(probsaL(1,:) + probsaL(2,:))*100;
plot(V3, ratio, 'k:', 'LineWidth', 1);
ratio = probsbL(1,:)./(probsbL(1,:) + probsbL(2,:))*100;
plot(V3, ratio, 'k-', 'LineWidth', 1);
xlabel('V3');
ylabel('% Correct (V1 & V2)');
xlim([0, V1mean]);
yticks = get(gca, 'YTick');
yticklabels = arrayfun(@(v) sprintf('%0.1f', v), yticks, 'UniformOutput', false);
set(gca, 'YTickLabel', yticklabels);
mysavefig(h, filename, plot_dir, 12, [6, 8]);