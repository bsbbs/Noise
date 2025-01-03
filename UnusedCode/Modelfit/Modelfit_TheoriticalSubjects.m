%% define directoriress
DefineIO;
% Switch to the working directory
cd(Fitdir);
%% Loading the data transformed in the code: /Users/bs3667/Noise/modelfit/ModelFit-DataTrnsfrm.m
load(fullfile(Gitdir, 'myData', 'TrnsfrmData.mat'), 'mt');
fitdir = fullfile(rootdir, 'Modelfit');
fit = tdfread(fullfile(fitdir, 'BestRslts.txt'));
plot_dir = fullfile(fitdir, 'plot');
%% Transform data
blacklist = [22102405; 22102705; 22102708; 22071913; 22110306];
sublist = unique(mt.subID);
sublist = sublist(~ismember(sublist, blacklist));
fulllist = unique(mt.subID);
N = length(sublist);
mtconvert = [];
for s = 1:N
    indvtask = mt(mt.subID == sublist(s),:);
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
% pairs = [mtconvert.V1scld, mtconvert.sdV1scld, mtconvert.V2scld, mtconvert.sdV2scld];
% unique_pairs = unique(pairs, 'rows'); % get the unique pairs of target values across all subjects for simulation below
V3scldp = unique([mtconvert.V3scld(mtconvert.Vaguenesscode == 0), mtconvert.sdV3scld(mtconvert.Vaguenesscode == 0)], 'rows'); % precise V3
V3scldv = unique([mtconvert.V3scld(mtconvert.Vaguenesscode == 1), mtconvert.sdV3scld(mtconvert.Vaguenesscode == 1)], 'rows'); % vague V3
%% Simulation
for modeli = 1:4
    simdat = fullfile(fitdir, sprintf('Model%i_PredictTheoriticalSubjects.mat', modeli));
    if ~exist(simdat, 'file')
        mtmodel = [];
        for s = 1:N
            fprintf('Subject %d:\t', s);
            % generate set of data the same for every individual
            dat = table('Size', [0 16], 'VariableTypes', {'int32', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double'},...
                'VariableNames', {'subID', 'V1', 'V2', 'V3', 'sdV1','sdV2', 'sdV3', 'V1scld', 'V2scld', 'V3scld', 'sdV1scld','sdV2scld', 'sdV3scld', 'Vaguenesscode', 'TimeConstraint', 'chosenItem'});
            indvtask = mt(mt.subID == sublist(s),:);
            pairs = [indvtask.V1, indvtask.sdV1, indvtask.V2, indvtask.sdV2];
            unique_pairs = unique(pairs, 'rows');
            mintrgt = min(unique([indvtask.V1; indvtask.V2]));
            % 'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3'
            V1 = unique_pairs(:,1);
            sdV1 = unique_pairs(:,2);
            V2 = unique_pairs(:,3);
            sdV2 = unique_pairs(:,4);
            V1scld = V1/mintrgt;
            sdV1scld = sdV1/mintrgt;
            V2scld = V2/mintrgt;
            sdV2scld = sdV2/mintrgt;
            
            trgt_idx = 1:numel(V1);

            for vi = 1:2 % precise and vague condition
                if vi == 1
                    V3scld = V3scldp(:,1);
                    sdV3scld = V3scldp(:,2);
                elseif vi == 2
                    V3scld = V3scldv(:,1);
                    sdV3scld = V3scldv(:,2);
                end
                V3 = V3scld*mintrgt;
                sdV3 = sdV3scld*mintrgt;
                V3_idx = 1:numel(V3);
                [ComboTrgt, ComboV3] = meshgrid(trgt_idx, V3_idx);
                for t = [10, 1.5]
                    Nrows = numel(ComboTrgt);
                    new_rows = table(repmat(sublist(s), Nrows, 1), ...
                        V1(ComboTrgt(:)), V2(ComboTrgt(:)), V3(ComboV3(:)), ...
                        sdV1(ComboTrgt(:)), sdV2(ComboTrgt(:)), sdV3(ComboV3(:)),...
                        V1scld(ComboTrgt(:)), V2scld(ComboTrgt(:)), V3scld(ComboV3(:)), ...
                        sdV1scld(ComboTrgt(:)), sdV2scld(ComboTrgt(:)), sdV3scld(ComboV3(:)),...
                        ones(Nrows, 1)*(vi-1), t*ones(Nrows, 1), nan(Nrows, 1), 'VariableNames', dat.Properties.VariableNames);
                    dat = [dat; new_rows];
                end
            end

            subjmask = fit.subID == sublist(s);
            Mp = fit.Mp(subjmask & fit.modeli == modeli);
            delta = fit.delta(subjmask & fit.modeli == modeli);
            switch modeli
                case 1
                    %%
                    x = [Mp, delta];
                    probs = McFadden(x, dat);
                    name = 'McFadden';
                case 2
                    scl = fit.scl(subjmask & fit.modeli == modeli);
                    x = [Mp, delta, scl];
                    probs = Mdl2(x, dat);
                    name = 'LinearDistrb';
                case 3
                    wp = fit.wp(subjmask & fit.modeli == modeli);
                    x = [Mp, delta, wp];
                    probs = DNM(x, dat);
                    name = 'DNM'; % classical divisive normalization model
                case 4
                    scl = fit.scl(subjmask & fit.modeli == modeli);
                    wp = fit.wp(subjmask & fit.modeli == modeli);
                    x = [Mp, delta, wp, scl];
                    probs = dnDNM(x, dat, 'absorb');
                    name = 'dnDNM'; % divisive normalization model with two stages of noise
            end
            dat.modelprob1 = probs(:,1);
            dat.modelprob2 = probs(:,2);
            dat.modelprob3 = probs(:,3);
            dat.ratio = dat.modelprob2./(dat.modelprob1 + dat.modelprob2);
%             IndvMean = grpstats(dat, ["TimeConstraint", "Vaguenesscode", "V3scld", "sdV3scld"], "mean", "DataVars", ["subID", "V3scld", "sdV3scld", "ratio"]);
%             IndvMean.Properties.RowNames = {};
            mtmodel = [mtmodel; dat];
            fprintf('\n');
        end
        save(simdat, "mtmodel", '-mat');
    else
        load(simdat);
    end
    %% Visualize in sliding windows
    GrpMean = grpstats(mtmodel, ["TimeConstraint", "Vaguenesscode", "V3scld"], "mean", "DataVars", ["V3scld", "sdV3scld", "ratio"]);
    colorpalette ={'r','#FFBF00','#00FF80','b'};
    rgbMatrix = [
        0, 0, 255;   % Blue
        255, 192, 203; % Pink
        173, 216, 230; % Light Blue
        255, 0, 0     % Red
        ]/255;
    Window = 0.03/2;
    LowestV3 = 0; %0.2;
    HighestV3 = 1; %.8;
    h = figure;
    filename = sprintf('Model%i_PredictTheoriticalSubjects', modeli);
    vi = 0;
    i = 0;
    for v = [1, 0] % vague, precise
        vi = vi + 1;
        subplot(1,2,vi); hold on;
        for t = [10, 1.5] % low, high
            i = i + 1;
            Ntrial = [];
            %choice = [];
            %choicese = [];
            ratio = [];
            ratiose = [];
            sdV3scld = [];
            v3vec = LowestV3:.03:HighestV3;
            dat = GrpMean(GrpMean.TimeConstraint == t & GrpMean.Vaguenesscode == v & GrpMean.mean_V3scld >= LowestV3 &  GrpMean.mean_V3scld <= HighestV3,:);
            for v3 = v3vec
                section = dat(dat.mean_V3scld >= v3 - Window & dat.mean_V3scld <= v3 + Window,:);
                Ntrial = [Ntrial, sum(section.GroupCount)];
                %choice = [choice, mean(section.mean_choice)];
                %choicese = [choicese, std(section.mean_choice)/sqrt(length(section.mean_choice))];
                ratio = [ratio, mean(section.mean_ratio)];
                ratiose = [ratiose, std(section.mean_ratio)/sqrt(length(section.mean_ratio))];
                sdV3scld = [sdV3scld, mean(section.mean_sdV3scld)];
            end
            cut = Ntrial > 400;
            % scatter(v3vec(cut), ratio(cut), Ntrial(cut)/80*5, 'color', colorpalette{i});
            % plot(v3vec(cut), choice(cut), '-', 'Color', colorpalette{i}, 'LineWidth', 2);
            plot(v3vec(cut), ratio(cut), 'k--','Color', colorpalette{i}, 'LineWidth', 2);
            % fill([v3vec fliplr(v3vec)], [ratio-ratiose fliplr(ratio+ratiose)], rgbMatrix(vi,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        end
        xlim([LowestV3, HighestV3]);
        xlabel('Scaled V3');
        ylabel('% Correct | V1, V2');
        mysavefig(h, filename, plot_dir, 12, [8, 4]);
    end
    %% Visualization in heatmap
    GrpMean = grpstats(mtmodel, ["TimeConstraint", "Vaguenesscode", "V3scld", "sdV3scld"], "mean", "DataVars", ["V3scld", "sdV3scld", "ratio"]);
    Window = 0.08/2;
    Varrng = [min(GrpMean.mean_sdV3scld), .8];% max(GrpMean.mean_sdV3scld)];
    Bindow = 0.08/2;
    h = figure;
    filename = sprintf('ModelFit_%i_HeatmapTheoriticalSubjects', modeli);
    ti = 0;
    TimePressure = {'Low','High'};
    for t = [10, 1.5] % low, high
        ti = ti + 1;
        dat = GrpMean(GrpMean.TimeConstraint == t,:);
        v3vec = LowestV3:.08:HighestV3;
        varvec = Varrng(1):.08:Varrng(2);
        Ntrial = NaN(numel(varvec), numel(v3vec));
        ratio = NaN(numel(varvec), numel(v3vec));
        ratiose = NaN(numel(varvec), numel(v3vec));
        sdV3scld = NaN(numel(varvec), numel(v3vec));
        for vi = 1:numel(v3vec)
            for ri = 1:numel(varvec)
                v3 = v3vec(vi);
                r = varvec(ri);
                maskv3 = dat.mean_V3scld >= v3 - Window & dat.mean_V3scld <= v3 + Window;
                maskr3 = dat.mean_sdV3scld >= r - Bindow & dat.mean_sdV3scld <= r + Bindow;
                section = dat(maskv3 & maskr3,:);
                Ntrial(ri,vi) = sum(section.GroupCount);
                ratio(ri,vi) = mean(section.mean_ratio);
                ratiose(ri,vi) = std(section.mean_ratio)/sqrt(length(section.mean_ratio));
                sdV3scld(ri,vi) = mean(section.mean_sdV3scld);
            end
        end
        ratio(Ntrial<50) = NaN;
        subplot(2, 2, 1+(ti-1)*2); hold on;
        colormap("bone");
        cmap = bone(numel(varvec));
        for ri = 1:numel(varvec)
            plot(v3vec, ratio(ri,:), '.-', 'Color', cmap(ri,:));
        end
        title(TimePressure{ti});
        xlabel('Scaled V3');
        ylabel('% Correct | V1 & V2');
        mysavefig(h, filename, plot_dir, 12, [9, 8]);
    
        subplot(2, 2, 2+(ti-1)*2); hold on;
        colormap("jet");
        imagesc(v3vec, varvec, ratio);
        c = colorbar('Location', 'northoutside');
        % ylim([0,1]);
        ylabel(c, '% Correct | V1 & V2');
        xlabel('Scaled V3');
        ylabel('V3 Variance');
        mysavefig(h, filename, plot_dir, 12, [9, 8]);
    end
end

%% 
function probs = McFadden(x, dat)
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1); % change M to be M', absorbing the magnitude of late noise
eta = 1; % after the transformation, the late noise term is standardized as 1
delta = x(2); % late noise difference between time-pressure conditions
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem','TimeConstraint'});
num_samples = 9000; %20000;
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
    else
        values = data.(['V',num2str(ci)])';
    end
    samples(:,:,ci) = repmat(values, num_samples, 1);
end
if gpuparallel
    SVs = samples/Mp + gpuArray.randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = samples/Mp + randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
tmp = squeeze(sum(max_from_each_distribution, 1));
probs = tmp ./ sum(tmp,2);
if gpuparallel
    probs = gather(probs);
end
end

function probs = Mdl2(x, dat)
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1); % change M to be M', absorbing the magnitude of late noise
eta = 1; % after the transformation, the late noise term is standardized as 1
delta = x(2); % late noise difference between time-pressure conditions
scl = x(3); % scaling parameter on early noise
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem','TimeConstraint'});
num_samples = 9000; %20000;
Ntrl = size(dat,1);
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
        stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
        samples(:,:,ci) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    else
        values = data.(['V',num2str(ci)])';
        stds = data.(['sdV', num2str(ci)])'*scl;
        samples(:,:,ci) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    end
end
if gpuparallel
    SVs = samples/Mp + gpuArray.randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = samples/Mp + randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
tmp = squeeze(sum(max_from_each_distribution, 1));
probs = tmp ./ sum(tmp,2);
if gpuparallel
    probs = gather(probs);
end
end

function probs = DNM(x, dat)
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1); % change M to be M', absorbing the magnitude of late noise
eta = 1; % after the transformation, the late noise term is standardized as 1
delta = x(2); % late noise difference between time-pressure conditions
wp = x(3); % do the same transformation on w
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem','TimeConstraint'});
num_samples = 9000; %20000;
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
    else
        values = data.(['V',num2str(ci)])';
    end
    samples(:,:,ci) = repmat(values, num_samples, 1);
end
% The product of divisive normalization before adding late noise
DNP = samples./(sum(samples, 3)*wp + Mp);
% adding late noise
if gpuparallel
    SVs = DNP + gpuArray.randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = DNP + randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
tmp = squeeze(sum(max_from_each_distribution, 1));
probs = tmp ./ sum(tmp,2);
if gpuparallel
    probs = gather(probs);
end
end

function probs = dnDNM(x, dat, mode) % cut inputs, independent
% set the lower boundary for every input value distribution as zero
% samples in the denominator are independent from the numerator
% the SIGMA term in the denominator is non-negative after that.
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1); % change M to be M', absorbing the magnitude of late noise
eta = 1; % after the transformation, the late noise term is standardized as 1
delta = x(2); % late noise difference between time-pressure conditions
wp = x(3); % do the same transformation on w
scl = x(4); % scaling parameter on the early noise
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem','TimeConstraint'});
num_samples = 9000; %20000;
Ntrl = size(dat,1);
if strcmp(mode, 'absorb')
    samples = [];
    for ci = 1:3
        if gpuparallel
            values = gpuArray(data.(['V',num2str(ci)])');
            stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
            samples(:,:,ci) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        else
            values = data.(['V',num2str(ci)])';
            stds = data.(['sdV', num2str(ci)])'*scl;
            samples(:,:,ci) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        end
    end
    D1 = [];
    D2 = [];
    D3 = [];
    for ci = 1:3
        if gpuparallel
            values = gpuArray(data.(['V',num2str(ci)])');
            stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
            D1(:,:,ci) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D2(:,:,ci) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D3(:,:,ci) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        else
            values = data.(['V',num2str(ci)])';
            stds = data.(['sdV', num2str(ci)])'*scl;
            D1(:,:,ci) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D2(:,:,ci) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D3(:,:,ci) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        end
    end
elseif strcmp(mode, 'cutoff')
    error('The cutoff boundary algorithm has not been developped yet.');
end
D1 = sum(D1, 3)*wp + Mp;
D2 = sum(D2, 3)*wp + Mp;
D3 = sum(D3, 3)*wp + Mp;
% The product of divisive normalization before adding late noise
DNP = samples./cat(3, D1, D2, D3);
clear D1 D2 D3;
if gpuparallel
    SVs = DNP + gpuArray.randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = DNP + randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
tmp = squeeze(sum(max_from_each_distribution, 1));
probs = tmp ./ sum(tmp,2);
if gpuparallel
    probs = gather(probs);
end
end

