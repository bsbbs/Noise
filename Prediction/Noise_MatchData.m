%% Simulation on noise that matches the value settings in the empirical data
%% defiine directories
Glgdir = '/Volumes/GoogleDrive/My Drive';
Gitdir = '/home/bo/Documents/Noise';
Gitdir = '~/Documents/Noise';
addpath(genpath(Gitdir));
plot_dir = fullfile(Glgdir, 'Noise','Prediction');
plot_dir = '/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/Prediction';
%% where is the data?
% datdir = '/gpfs/data/glimcherlab/BoShen/Noise/CESS-Bo_Nov2022/TaskProgram/log/txtDat';
datdir = '/Volumes/BoShen/Noise/CESS-Bo_Nov2022/TaskProgram/log/txtDat';
filelist = dir(fullfile(datdir, 'MainTask*.txt'));
N = numel(filelist);
VtrgtPool = nan(N*360*2,1);
V3_scl2minPool = nan(N*360,1);
Vtrgt_scl2minPool = nan(N*360*2,1);
V3_scl2maxPool = nan(N*360,1);
Vtrgt_scl2maxPool = nan(N*360*2,1);
V1_scl2min = nan(N*360,1);
V2_scl2min = nan(N*360,1);
V3_scl2min = nan(N*360,1);
V3Pool = nan(N*360,1);
for s = 1:N
    indvtask = tdfread(fullfile(filelist(s).folder,filelist(s).name));
    VtrgtPool(((s-1)*720+1):(s*720)) = [indvtask.V1; indvtask.V2];
    V3Pool(((s-1)*360+1):(s*360)) = indvtask.V3;
    
    Vtrgt = unique([indvtask.V1; indvtask.V2]);
    V3 = unique(indvtask.V3);
    maxV = max([Vtrgt; V3]);
    mintrgt = min(Vtrgt);

    Vtrgt_scl2minPool(((s-1)*720+1):(s*720)) = [indvtask.V1; indvtask.V2]/mintrgt;
    Vtrgt_scl2maxPool(((s-1)*720+1):(s*720)) = [indvtask.V1; indvtask.V2]/maxV;
    V3_scl2minPool(((s-1)*360+1):(s*360)) = indvtask.V3/mintrgt;
    V3_scl2maxPool(((s-1)*360+1):(s*360)) = indvtask.V3/maxV;
    V1_scl2min(((s-1)*360+1):(s*360)) = indvtask.V1/mintrgt;
    V2_scl2min(((s-1)*360+1):(s*360)) = indvtask.V2/mintrgt;
    V3_scl2min(((s-1)*360+1):(s*360)) = indvtask.V3/mintrgt;
end
%% visualize the empirical value distributions
filename = 'ExpDsgn_Vdstrb_scl2min';
h = Pltdstrbtn(Vtrgt_scl2minPool, V3_scl2minPool, plot_dir, filename);
filename = 'ExpDsgn_Vdstrb_scl2max';
h = Pltdstrbtn(Vtrgt_scl2maxPool, V3_scl2maxPool, plot_dir, filename);


%% setting of input values
valid = V1_scl2min < Inf & ~isnan(V1_scl2min) & V2_scl2min < Inf & ~isnan(V2_scl2min);
V1 = V1_scl2min(valid);
V2 = V2_scl2min(valid);
V3 = V3_scl2min(valid);
M = mean([V1; V2; V3]);
V3 = 0:.05:1;
NV3 = numel(V3);
plot(V1,V2,'.');
%% Representational noise only
sgm1 = 1;
sgm2 = 1;
sgm3 = 1;
eta1 = 0;
eta2 = 0;
eta3 = 0;
h = figure;
fontsize = 11;
aspect = [12, 8];
lwd = 2;
filename = '[MatchData]CR_RprNoise';
method = {'Dpdnt', 'Indpdnt'};
for mi = 1:2
    subplot(2,3,1+(mi-1)*3);
    hold on;
    Ntrials = nan([1000,numel(V3)]);
    accuracy = nan([1000,numel(V3)]);
    CutoffN = nan([1000,numel(V3)]);
    CutoffD = nan([1000,numel(V3)]);
    CutoffV = nan([1000,numel(V3)]);
    for i = 1:1000
        for v = 1:numel(V3)
            [SV1, SV2, SV3, cutoffN, cutoffD, cutoffV] = ...
                eval(['DVN' method{mi} '(V1, sgm1, eta1, V2, sgm2, eta2,  V3(v)*ones(size(V1)), sgm3, eta3, M);']);
            conditional = SV3 < SV1 & SV3 < SV2;
            choice =  SV2 > SV1;
            Ntrials(i,v) = sum(conditional);
            accuracy(i,v) = mean(choice(conditional), 'omitnan');
            CutoffN(i,v) = cutoffN/numel(SV1);
            CutoffD(i,v) = cutoffD/numel(SV1);
            CutoffV(i,v) = cutoffV/numel(SV1);
        end
        if i == 1000
            plot(V3, accuracy(i,:), '.-', 'LineWidth', lwd+1, 'Color', colorpalette{2});
        else
            lp = plot(V3, accuracy(i,:), '.-');
        end
    end
    xlabel('V_3','FontAngle','italic');
    ylabel('Accuracy|V_1, V_2','FontAngle','italic');
    title('Single sets', sprintf('%s Noise', method{mi}));
    mysavefig(h, filename, plot_dir, fontsize, aspect);
    subplot(2,3,2+(mi-1)*3);
    hold on;
    plot(V3, mean(accuracy,1), 'b.-', 'LineWidth', lwd);
    scatter(V3, mean(accuracy,1), mean(Ntrials, 1)/max(mean(Ntrials,1))*200, 'b');
    xlabel('V_3','FontAngle','italic');
    ylabel('Accuracy|V_1, V_2','FontAngle','italic');
    title('1000-simulation average', sprintf('%s Noise', method{mi}));
    mysavefig(h, filename, plot_dir, fontsize, aspect);
    subplot(2,3,3+(mi-1)*3);
    hold on;
    lgd1 = plot(V3, mean(CutoffN,1)*100, '.-', 'Color', colorpalette{3}, 'LineWidth', lwd);
    lgd2 = plot(V3, mean(CutoffD,1)*100, '.-', 'Color', colorpalette{5}, 'LineWidth', lwd);
    lgd3 = plot(V3, mean(CutoffV,1)*100, '.-', 'Color', colorpalette{1}, 'LineWidth', lwd);
    if mi == 1
        legend([lgd1, lgd2, lgd3], {'Numerator','Denominator','Normalized values'}, 'Location', 'best');
    end
    xlabel('V_3','FontAngle','italic');
    ylabel('Cutoff occurs (% total trials)','FontAngle','italic');
    title('Cutoffs at zero', sprintf('%s Noise', method{mi}));
    mysavefig(h, filename, plot_dir, fontsize, aspect);
end

%% Decision noise only
sgm1 = 1*0;
sgm2 = 1*0;
sgm3 = 1*0;
eta1 = .1;
eta2 = .1;
eta3 = .1;
h = figure;
fontsize = 11;
aspect = [12, 8];
lwd = 2;
colorpalette = {'#ef476f','#ffd166','#06d6a0','#118ab2','#073b4c'};
filename = '[MatchData]CR_DcsnNoise';
method = {'Dpdnt', 'Indpdnt'};
for mi = 1:2
    subplot(2,3,1+(mi-1)*3);
    hold on;
    Ntrials = nan([1000,numel(V3)]);
    accuracy = nan([1000,numel(V3)]);
    CutoffN = nan([1000,numel(V3)]);
    CutoffD = nan([1000,numel(V3)]);
    CutoffV = nan([1000,numel(V3)]);
    for i = 1:1000
        for v = 1:numel(V3)
            [SV1, SV2, SV3, cutoffN, cutoffD, cutoffV] = ...
                eval(['DVN' method{mi} '(V1, sgm1, eta1, V2, sgm2, eta2,  V3(v)*ones(size(V1)), sgm3, eta3, M);']);
            conditional = SV3 < SV1 & SV3 < SV2;
            choice =  SV2 > SV1;
            Ntrials(i,v) = sum(conditional);
            accuracy(i,v) = mean(choice(conditional), 'omitnan');
            CutoffN(i,v) = cutoffN/numel(SV1);
            CutoffD(i,v) = cutoffD/numel(SV1);
            CutoffV(i,v) = cutoffV/numel(SV1);
        end
        if i == 1000
            plot(V3, accuracy(i,:), '.-', 'LineWidth', lwd+1, 'Color', colorpalette{2});
        else
            lp = plot(V3, accuracy(i,:), '.-');
        end
    end
    xlabel('V_3','FontAngle','italic');
    ylabel('Accuracy|V_1, V_2','FontAngle','italic');
    title('Single sets', sprintf('%s Noise', method{mi}));
    mysavefig(h, filename, plot_dir, fontsize, aspect);
    subplot(2,3,2+(mi-1)*3);
    hold on;
    plot(V3, mean(accuracy,1), 'b.-', 'LineWidth', lwd);
    scatter(V3, mean(accuracy,1), mean(Ntrials, 1)/max(mean(Ntrials,1))*200, 'b');
    xlabel('V_3','FontAngle','italic');
    ylabel('Accuracy|V_1, V_2','FontAngle','italic');
    title('1000-simulation average', sprintf('%s Noise', method{mi}));
    mysavefig(h, filename, plot_dir, fontsize, aspect);
    subplot(2,3,3+(mi-1)*3);
    hold on;
    lgd1 = plot(V3, mean(CutoffN,1)*100, '.-', 'Color', colorpalette{3}, 'LineWidth', lwd);
    lgd2 = plot(V3, mean(CutoffD,1)*100, '.-', 'Color', colorpalette{5}, 'LineWidth', lwd);
    lgd3 = plot(V3, mean(CutoffV,1)*100, '.-', 'Color', colorpalette{1}, 'LineWidth', lwd);
    if mi == 1
        legend([lgd1, lgd2, lgd3], {'Numerator','Denominator','Normalized values'}, 'Location', 'best');
    end
    xlabel('V_3','FontAngle','italic');
    ylabel('Cutoff occurs (% total trials)','FontAngle','italic');
    title('Cutoffs at zero', sprintf('%s Noise', method{mi}));
    mysavefig(h, filename, plot_dir, fontsize, aspect);
end
%% Mixed noise
sgm1 = 1;
sgm2 = 1;
sgm3 = 1;
eta1 = .1;
eta2 = .1;
eta3 = .1;
h = figure;
fontsize = 11;
aspect = [12, 8];
lwd = 2;
colorpalette = {'#ef476f','#ffd166','#06d6a0','#118ab2','#073b4c'};
filename = '[MatchData]CR_MixNoise';
method = {'Dpdnt', 'Indpdnt'};
for mi = 1:2
    subplot(2,3,1+(mi-1)*3);
    hold on;
    Ntrials = nan([1000,numel(V3)]);
    accuracy = nan([1000,numel(V3)]);
    CutoffN = nan([1000,numel(V3)]);
    CutoffD = nan([1000,numel(V3)]);
    CutoffV = nan([1000,numel(V3)]);
    for i = 1:1000
        for v = 1:numel(V3)
            [SV1, SV2, SV3, cutoffN, cutoffD, cutoffV] = ...
                eval(['DVN' method{mi} '(V1, sgm1, eta1, V2, sgm2, eta2,  V3(v)*ones(size(V1)), sgm3, eta3, M);']);
            conditional = SV3 < SV1 & SV3 < SV2;
            choice =  SV2 > SV1;
            Ntrials(i,v) = sum(conditional);
            accuracy(i,v) = mean(choice(conditional), 'omitnan');
            CutoffN(i,v) = cutoffN/numel(SV1);
            CutoffD(i,v) = cutoffD/numel(SV1);
            CutoffV(i,v) = cutoffV/numel(SV1);
        end
        if i == 1000
            plot(V3, accuracy(i,:), '.-', 'LineWidth', lwd+1, 'Color', colorpalette{2});
        else
            lp = plot(V3, accuracy(i,:), '.-');
        end
    end
    xlabel('V_3','FontAngle','italic');
    ylabel('Accuracy|V_1, V_2','FontAngle','italic');
    title('Single sets', sprintf('%s Noise', method{mi}));
    mysavefig(h, filename, plot_dir, fontsize, aspect);
    subplot(2,3,2+(mi-1)*3);
    hold on;
    plot(V3, mean(accuracy,1), 'b.-', 'LineWidth', lwd);
    scatter(V3, mean(accuracy,1), mean(Ntrials, 1)/max(mean(Ntrials,1))*200, 'b');
    xlabel('V_3','FontAngle','italic');
    ylabel('Accuracy|V_1, V_2','FontAngle','italic');
    title('1000-simulation average', sprintf('%s Noise', method{mi}));
    mysavefig(h, filename, plot_dir, fontsize, aspect);
    subplot(2,3,3+(mi-1)*3);
    hold on;
    lgd1 = plot(V3, mean(CutoffN,1)*100, '.-', 'Color', colorpalette{3}, 'LineWidth', lwd);
    lgd2 = plot(V3, mean(CutoffD,1)*100, '.-', 'Color', colorpalette{5}, 'LineWidth', lwd);
    lgd3 = plot(V3, mean(CutoffV,1)*100, '.-', 'Color', colorpalette{1}, 'LineWidth', lwd);
    if mi == 1
        legend([lgd1, lgd2, lgd3], {'Numerator','Denominator','Normalized values'}, 'Location', 'best');
    end
    xlabel('V_3','FontAngle','italic');
    ylabel('Cutoff occurs (% total trials)','FontAngle','italic');
    title('Cutoffs at zero', sprintf('%s Noise', method{mi}));
    mysavefig(h, filename, plot_dir, fontsize, aspect);
end

%% Mixed noise - multiple levels
Tunes = linspace(0,1,8);
h = figure;
fontsize = 11;
aspect = [4, 8];
lwd = 2;
mycols = [winter(4); flip(autumn(4))];
filename = sprintf('[MatchData]CR_MixNoise_%ilevels', numel(Tunes));
method = {'Dpdnt', 'Indpdnt'};
for mi = 1:2
    subplot(2,1,mi);
    hold on;
    ldg = [];
    for ti = 1:numel(Tunes)
        shrd = .85*Tunes(ti); % 1, .75, .85
        sgm1 = shrd;
        sgm2 = shrd;
        sgm3 = shrd;
        ehrd = .13*(1-Tunes(ti)); % .1, .15, 1.3
        eta1 = ehrd;
        eta2 = ehrd;
        eta3 = ehrd;
        
        Ntrials = nan([1000,numel(V3)]);
        accuracy = nan([1000,numel(V3)]);
        CutoffN = nan([1000,numel(V3)]);
        CutoffD = nan([1000,numel(V3)]);
        CutoffV = nan([1000,numel(V3)]);
        for v = 1:NV3
            V3i = V3(v)*ones(size(V1));
            parfor i = 1:1000
                SV1 = nan(size(V1));
                SV2 = nan(size(V1));
                SV3 = nan(size(V1));
                cutoffN = nan(size(V1));
                cutoffD = nan(size(V1));
                cutoffV = nan(size(V1));
                switch mi
                    case 1
                        [SV1, SV2, SV3, cutoffN, cutoffD, cutoffV] = ...
                            DVNDpdnt(V1, sgm1, eta1, V2, sgm2, eta2,  V3i, sgm3, eta3, M);
                    case 2
                        [SV1, SV2, SV3, cutoffN, cutoffD, cutoffV] = ...
                            DVNIndpdnt(V1, sgm1, eta1, V2, sgm2, eta2,  V3i, sgm3, eta3, M);
                end
                conditional = SV3 < SV1 & SV3 < SV2;
                choice =  SV2 > SV1;
                Ntrials(i,v) = sum(conditional);
                accuracy(i,v) = mean(choice(conditional), 'omitnan');
                CutoffN(i,v) = cutoffN/numel(SV1);
                CutoffD(i,v) = cutoffD/numel(SV1);
                CutoffV(i,v) = cutoffV/numel(SV1);
            end
        end
        ldg(ti) = plot(V3, mean(accuracy,1), '.-', 'LineWidth', lwd, 'Color',mycols(ti,:));
    end
    legend(ldg, {'','','','','','','',''},...
    'FontSize', 7, 'Location', 'eastoutside');
    xlabel('V_3','FontAngle','italic');
    ylabel('Accuracy|V_1, V_2','FontAngle','italic');
    title('1000-simulation average', sprintf('%s Noise', method{mi}));
    mysavefig(h, filename, plot_dir, fontsize, aspect);
end


%% functions
function [SV1, SV2, SV3, cutoffN, cutoffD, cutoffV] = DVNDpdnt(V1, sgm1, eta1, V2, sgm2, eta2,  V3, sgm3, eta3, M)

s1 = sgm1*randn(size(V1));
s2 = sgm2*randn(size(V2));
s3 = sgm3*randn(size(V3));
e1 = eta1*randn(size(V1));
e2 = eta2*randn(size(V2));
e3 = eta3*randn(size(V3));
Denominator = (M + V1 + s1 + V2 + s2 + V3 + s3);
cutoffD = Denominator<0;
Denominator(cutoffD) = 0;
cutoffD = sum(cutoffD);

Numerator = (V1 + s1);
cutoffN1 = Numerator<0;
Numerator(cutoffN1) = 0;
SV1 = Numerator./Denominator + e1;
cutoffV1 = SV1 < 0;
SV1(cutoffV1) = 0;

Numerator = (V2 + s2);
cutoffN2 = Numerator<0;
Numerator(cutoffN2) = 0;
SV2 = Numerator./Denominator + e2;
cutoffV2 = SV2 < 0;
SV2(cutoffV2) = 0;

Numerator = (V3 + s3);
cutoffN3 = Numerator<0;
Numerator(cutoffN3) = 0;
SV3 = Numerator./Denominator + e3;
cutoffV3 = SV3 < 0;
SV3(cutoffV3) = 0;
cutoffN = sum(cutoffN1 | cutoffN2 | cutoffN3);
cutoffV = sum(cutoffV1 | cutoffV2 | cutoffV3);
end

function [SV1, SV2, SV3, cutoffN, cutoffD, cutoffV] = DVNIndpdnt(V1, sgm1, eta1, V2, sgm2, eta2,  V3, sgm3, eta3, M)

e1 = eta1*randn(size(V1));
e2 = eta2*randn(size(V2));
e3 = eta3*randn(size(V3));

Denominator1 = (M + V1 + sgm1*randn(size(V1)) + V2 + sgm2*randn(size(V2)) + V3 + sgm3*randn(size(V3)));
Denominator2 = (M + V1 + sgm1*randn(size(V1)) + V2 + sgm2*randn(size(V2)) + V3 + sgm3*randn(size(V3)));
Denominator3 = (M + V1 + sgm1*randn(size(V1)) + V2 + sgm2*randn(size(V2)) + V3 + sgm3*randn(size(V3)));
cutoffD1 = Denominator1<0;
Denominator1(cutoffD1) = 0;
cutoffD2 = Denominator2<0;
Denominator2(cutoffD2) = 0;
cutoffD3 =  Denominator3<0;
Denominator3(cutoffD3) = 0;
cutoffD = sum(cutoffD1 | cutoffD2 | cutoffD3);

Numerator1 = (V1 + sgm1*randn(size(V1)));
cutoffN1 = Numerator1<0;
Numerator1(cutoffN1) = 0;
SV1 = Numerator1./Denominator1 + e1;
cutoffV1 = SV1 < 0;
SV1(cutoffV1) = 0;

Numerator2 = (V2 + sgm2*randn(size(V2)));
cutoffN2 = Numerator2<0;
Numerator2(cutoffN2) = 0;
SV2 = Numerator2./Denominator2 + e2;
cutoffV2 = SV2 < 0;
SV2(cutoffV2) = 0;

Numerator3 = (V3 + sgm3*randn(size(V3)));
cutoffN3 = Numerator3<0;
Numerator3(cutoffN3) = 0;
SV3 = Numerator3./Denominator3 + e3;
cutoffV3 = SV3 < 0;
SV3(cutoffV3) = 0;
cutoffN = sum(cutoffN1 | cutoffN2 | cutoffN3);
cutoffV = sum(cutoffV1 | cutoffV2 | cutoffV3);
end