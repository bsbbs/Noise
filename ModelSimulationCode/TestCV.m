%% define directories
[os, ~, ~] = computer;
if os == 'MACI64'
    rootdir = '/Users/bs3667/Dropbox (NYU Langone Health)/';
end
plot_dir = fullfile(rootdir, 'Bo Shen Working files/NoiseProject/Prediction');
dumpdir = plot_dir;
Gitdir = '~/Documents/Noise';
addpath(genpath(Gitdir));
%% Test the coefficient of variance
x = 10;
V1 = x + randn(1e5,1,'double');
filename = sprintf('CoefVarTest');
h = figure;
subplot(4,2,1); hold on;
histogram(V1/1, 'FaceColor', mycolors(1,:), EdgeColor = 'none');
histogram(V1/10, 'FaceColor', mycolors(end,:), EdgeColor = 'none');
ylabel('Freq.');
xlabel('V1 output');

z = 1:10;
mycolors = ones(1,3).*linspace(0.5,0,numel(z))';
CV = [];
for i = 1:numel(z)
    R1 = V1./z(i);
    CV(i) = std(R1)/mean(R1);
end
subplot(4,2,2);
scatter(z, CV, [], mycolors, 'filled');
% plot(z, CV, '-');
% Set Y-axis tick positions
yticks = get(gca, 'YTick');

% Format Y-axis tick labels with desired precision
% For example, to display with two decimal places
yticklabels = arrayfun(@(v) sprintf('%0.2f', v), yticks, 'UniformOutput', false);

% Apply the formatted labels to the plot
set(gca, 'YTickLabel', yticklabels);
ylabel('CoV');
xlabel('Constant denominator');
title('V1/Constant D');
% mysavefig(h, sprintf('CoefVarTest'), plot_dir, 12, [9, 4]);
%
subplot(4,2,3); hold on;
histogram(V1./(V1 + 1), 'FaceColor', mycolors(1,:), EdgeColor = 'none');
histogram(V1./(V1 + 10), 'FaceColor', mycolors(end,:), EdgeColor = 'none');
ylabel('Freq.');
xlabel('V1 output');

z = 1:10;
mycolors = ones(1,3).*linspace(0.5,0,numel(z))';
CV = [];
for i = 1:numel(z)
    R1 = V1./(V1 + z(i));
    CV(i) = std(R1)/mean(R1);
end
subplot(4,2,4);
scatter(z, CV, [], mycolors, 'filled');
%plot(z, CV, '.-', 'Color', mycolors);
% Set Y-axis tick positions
yticks = get(gca, 'YTick');

% Format Y-axis tick labels with desired precision
% For example, to display with two decimal places
yticklabels = arrayfun(@(v) sprintf('%0.4f', v), yticks, 'UniformOutput', false);

% Apply the formatted labels to the plot
set(gca, 'YTickLabel', yticklabels);
ylabel('CoV');
xlabel('Constant denominator');
title('V1/(V1 + Constant D)');
% mysavefig(h, sprintf('CoefVarTest'), plot_dir, 12, [9, 4]);

% Under the case of divisive normalization
subplot(4,2,5); hold on;
histogram(V1./(V1 + 1 + randn(1e5,1,'double')), 'FaceColor', mycolors(1,:), EdgeColor = 'none');
histogram(V1./(V1 + 10 + randn(1e5,1,'double')), 'FaceColor', mycolors(end,:), EdgeColor = 'none');
ylabel('Freq.');
xlabel('V1 output');

z = 1:10;
sgm = 0;
CV = [];
for i = 1:numel(z)
    R1 = V1./(sgm + V1 + z(i) + randn(1e5,1,'double'));
    CV(i) = std(R1)/mean(R1);
end
subplot(4,2,6);
scatter(z, CV, [], mycolors, 'filled');
% plot(z, CV, '-');
% Set Y-axis tick positions
yticks = get(gca, 'YTick');

% Format Y-axis tick labels with desired precision
% For example, to display with two decimal places
yticklabels = arrayfun(@(v) sprintf('%0.2f', v), yticks, 'UniformOutput', false);

% Apply the formatted labels to the plot
set(gca, 'YTickLabel', yticklabels);
ylabel('CoV');
xlabel('V3 mean');
title('V1/(V1 + V3)');
% mysavefig(h, sprintf('CoefVarTest'), plot_dir, 12, [5, 8]);
%
subplot(4,2,7); hold on;
V2 = x + randn(1e5,1,'double');
histogram(V1./(V2 + 1), 'FaceColor', mycolors(1,:), EdgeColor = 'none');
histogram(V1./(V2 + 10), 'FaceColor', mycolors(end,:), EdgeColor = 'none');
ylabel('Freq.');
xlabel('V1 output');

z = 1:10;
sgm = 0;
CV = [];
for i = 1:numel(z)
    R1 = V1./(sgm + V2 + z(i));
    CV(i) = std(R1)/mean(R1);
end
subplot(4,2,8);
scatter(z, CV, [], mycolors, 'filled');
% plot(z, CV, '-');
% Set Y-axis tick positions
yticks = get(gca, 'YTick');

% Format Y-axis tick labels with desired precision
% For example, to display with two decimal places
yticklabels = arrayfun(@(v) sprintf('%0.2f', v), yticks, 'UniformOutput', false);

% Apply the formatted labels to the plot
set(gca, 'YTickLabel', yticklabels);
ylabel('CoV');
xlabel('Constant denominator');
title('V1/(V2 + Constant D)');

h.PaperUnits = 'inches';
h.PaperPosition = [0 0 5, 8];
saveas(h,fullfile(plot_dir,sprintf('%s.pdf',filename)),'pdf');

%mysavefig(h, sprintf('CoefVarTest'), plot_dir, 12, [5, 8]);