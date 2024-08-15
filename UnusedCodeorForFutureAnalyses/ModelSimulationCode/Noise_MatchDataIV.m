%% Transform data
Sublist = unique(mt.subID);
N = length(Sublist);
mtconvert = [];
for s = 1:N
    indvtask = mt(mt.subID == Sublist(s),:);
    Vtrgt = unique([indvtask.V1; indvtask.V2]);
    mintrgt = min(Vtrgt);
    if mintrgt > 0 % skip this subject if the min target value is zero, because the values cannot be scaled and the value space does not help in testing of the hypothesis
        indvtask.V1scld = indvtask.V1/mintrgt;
        indvtask.V2scld = indvtask.V2/mintrgt;
        indvtask.V3scld = indvtask.V3/mintrgt;
        indvtask.sdV1scld = indvtask.sdV1/mintrgt;
        indvtask.sdV2scld = indvtask.sdV2/mintrgt;
        indvtask.sdV3scld = indvtask.sdV3/mintrgt;
        mtconvert = [mtconvert; indvtask];
    end  
end
mtconvert.choice = mtconvert.chosenItem - 1;

%% Choice accuracy in individuals
dat = mtconvert(mtconvert.chosenItem ~= 3 & ~isnan(mtconvert.chosenItem),:);
GrpMeanraw = grpstats(dat, ["subID","TimeConstraint", "Vaguenesscode", "ID3"], "mean", "DataVars", ["V3", "sdV3", "V3scld", "sdV3scld", "choice"]);
Treatment = 'Raw'; %'Demean'; %
LowestV3 = 0;
Sublist = unique(GrpMeanraw.subID);
N = length(Sublist);
if strcmp(Treatment, "Demean")
GrpMean = [];
for s = 1:N
    indv = GrpMeanraw(GrpMeanraw.subID == Sublist(s),:);
    indv.mean_choice = indv.mean_choice - mean(indv.mean_choice);
    GrpMean = [GrpMean; indv];
end
elseif strcmp(Treatment, 'Raw')
    GrpMean = GrpMeanraw;
end
%% Choice accuracy in a heatmap of mean value and variance
Window = 1.5;
Varrng = [min(GrpMean.mean_sdV3), 10];% max(GrpMean.mean_sdV3scld)];
Bindow = 1.5;
h = figure;
ti = 0;
for t = [10, 1.5] % low, high
    ti = ti + 1;
    dat = GrpMean(GrpMean.mean_V3scld <= 1 & GrpMean.TimeConstraint == t & GrpMean.Vaguenesscode == 0,:);
    v3vec = LowestV3:.15:max(dat.mean_V3);
    varvec = Varrng(1):.15:Varrng(2);
    Ntrial = NaN(numel(varvec), numel(v3vec));
    choice = NaN(numel(varvec), numel(v3vec));
    choicese = NaN(numel(varvec), numel(v3vec));
    sdV3scld = NaN(numel(varvec), numel(v3vec));
    for vi = 1:numel(v3vec)
        for ri = 1:numel(varvec)
            v3 = v3vec(vi);
            r = varvec(ri);
            maskv3 = dat.mean_V3 >= v3 - Window & dat.mean_V3 <= v3 + Window;
            maskr3 = dat.mean_sdV3 >= r - Bindow & dat.mean_sdV3 <= r + Bindow;
            section = dat(maskv3 & maskr3,:);
            Ntrial(ri,vi) = sum(section.GroupCount);
            choice(ri,vi) = mean(section.mean_choice);
            choicese(ri,vi) = std(section.mean_choice)/sqrt(length(section.mean_choice));
            sdV3scld(ri,vi) = mean(section.mean_sdV3);
        end
    end

    choice(Ntrial<30) = NaN;
    subplot(2,2,1+(ti-1)*2); hold on;
    colormap("bone");
    cmap = bone(numel(varvec));
    for ri = 1:numel(varvec)
        plot(v3vec, choice(ri,:), '.-', 'Color', cmap(ri,:));
    end
    xlabel('Scaled V3');
    ylabel('% Correct (V1 & V2)');
    %ylim([.4, .8]);
    title(sprintf('Time limit %1.1fs', t));

    subplot(2,2,2+(ti-1)*2);
    colormap("hot");
    imagesc(v3vec, varvec, choice, [0.4, 0.8]);
    title('');
    colorbar;
    xlabel('Scaled V3');
    ylabel('V3 Variance');
end
mysavefig(h, sprintf('ChoiceData_V3_%s_Precise', Treatment), plot_dir, 12, [10, 8]);


%% Individual difference on baseline accuracy
Sublist = unique(GrpMean.subID);
N = numel(Sublist);
h = figure; hold on;
for s = 1:N
    indv = GrpMean(GrpMean.subID == Sublist(s),:);
    plot3(indv.mean_V3scld, indv.mean_sdV3scld, indv.mean_choice,'.');
end
xlim([0,1]);
xlabel('Scaled V3');
ylabel("Var scaled V3");
zlabel('% Correct (V1 & V2)');

v3vec = 0:.015:1;
Window = 