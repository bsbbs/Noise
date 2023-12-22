%% Visualize in lines plot in x-axis of mean scaled V3
colorpalette = {'#0000FF', '#FFC0CB', '#ADD8E6', '#FF0000'};
rgbMatrix = [
    0, 0, 255;   % Blue
    255, 192, 203; % Pink
    173, 216, 230; % Light Blue
    255, 0, 0     % Red
]/255;
Window = 0.15;
h = figure;
hold on;
vi = 0;
Sldwndw = [];
for t = [10, 1.5] % low, high
    for v = [1, 0] % vague, precise
        vi = vi + 1;
        dat = GrpMean(GrpMean.TimeConstraint == t & GrpMean.Vaguenesscode == v,:);
        % plot(dat.mean_V3scld, dat.mean_choice, '.', 'Color', colorpalette{vi});
        Ntrial = [];
        choice = [];
        choicese = [];
        sdV3scld = [];
        v3vec = LowestV3:.015:1;
        for v3 = v3vec
            section = dat(dat.mean_V3scld >= v3 - Window & dat.mean_V3scld <= v3 + Window,:);
            Ntrial = [Ntrial, sum(section.GroupCount)];
            choice = [choice, mean(section.mean_choice)];
            choicese = [choicese, std(section.mean_choice)/sqrt(length(section.mean_choice))];
            sdV3scld = [sdV3scld, mean(section.mean_sdV3scld)];
        end
        tmp = [];
        tmp.TimeConstraint = t*ones(numel(Ntrial),1);
        tmp.Vaguenesscode = v*ones(numel(Ntrial),1);
        tmp.Ntrial = Ntrial';
        tmp.choice = choice';
        tmp.choicese = choicese';
        tmp.V3scld = v3vec';
        tmp.sdV3scld = sdV3scld';
        tmp = struct2table(tmp);
        Sldwndw = [Sldwndw; tmp];
        plot(v3vec, choice, '-', 'Color', colorpalette{vi}, 'LineWidth', 2);
        fill([v3vec fliplr(v3vec)], [choice-choicese fliplr(choice+choicese)], rgbMatrix(vi,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
end

xlim([0,1]);
xlabel('Scaled V3');
ylabel('% Correct (V1 & V2)');
mysavefig(h, sprintf('Ratio_Data_%s', Treatment), plot_dir, 12, [4, 4]);

%% Simulations - based on the trends of V3 noise
M = .1;
w = 1;
sL = .0000;
lL = .05;
eps = .1;
mode = 'absorb';
mtmodel = [];
for t = [10, 1.5] % low, high
    if t == 10
        x = [M, 1, sL];
    elseif t == 1.5
        x = [M, 1, lL];
    end
    for v = [1, 0] % vague, precise
        sect = Sldwndw(Sldwndw.TimeConstraint == t & Sldwndw.Vaguenesscode == v,:);
        dat = sect(:,{'TimeConstraint','Vaguenesscode','V3scld', 'sdV3scld', 'choice'});
        dat.Properties.VariableNames = {'TimeConstraint','Vaguenesscode','V3', 'sdV3', 'chosenItem'};
        dat.V1 = 1.05*ones(size(dat.V3));
        dat.V2 = 1.1533*ones(size(dat.V3));
        dat.sdV1 = eps*dat.V1;
        dat.sdV2 = eps*dat.V2;
        probs = dDN(x, dat, mode);
        ratio = probs(2,:)./sum(probs(1:2,:),1);
        dat.modelprob1 = probs(1,:)';
        dat.modelprob2 = probs(2,:)';
        dat.modelprob3 = probs(3,:)';
        mtmodel = [mtmodel; dat];
    end
end
mtmodel.ratio = mtmodel.modelprob2./(mtmodel.modelprob1 + mtmodel.modelprob2);

%
h = figure;
hold on;
vi = 0;
for t = [10, 1.5] % low, high
    for v = [1, 0] % vague, precise
        vi = vi + 1;
        dat = mtmodel(mtmodel.TimeConstraint == t & mtmodel.Vaguenesscode == v,:);
        plot(dat.V3, dat.ratio, '-', 'Color', colorpalette{vi}, 'LineWidth', 2);
        %fill([dat.V3 fliplr(dat.V3)], [ratio-ratiose fliplr(ratio+ratiose)], rgbMatrix(vi,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
end
xlim([0,1]);
xlabel('Scaled V3');
ylabel('% Correct (V1 & V2)');
mysavefig(h, sprintf('Ratio_ModelPredict_MatchOnlyV3_%s', Treatment), plot_dir, 12, [4, 4]);

%% functions
function probs = dDN(x, dat, mode) % cut inputs, independent
% set the lower boundary for every input value distribution as zero
% samples in the denominator are independent from the numerator
% the SIGMA term in the denominator will be natually non-negative after that.
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1);
wp = x(2);
eta = x(3);
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem'});
num_samples = 2000;
Ntrl = size(dat,1);
if strcmp(mode, 'absorb')
    samples = [];
    for ci = 1:3
        if gpuparallel
            values = gpuArray(data.(['V',num2str(ci)])');
            stds = gpuArray(data.(['sdV', num2str(ci)])');
            samples(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        else
            values = data.(['V',num2str(ci)])';
            stds = data.(['sdV', num2str(ci)])';
            samples(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        end
    end
    samplesD1 = [];
    samplesD2 = [];
    samplesD3 = [];
    for ci = 1:3
        if gpuparallel
            values = gpuArray(data.(['V',num2str(ci)])');
            stds = gpuArray(data.(['sdV', num2str(ci)])');
            samplesD1(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            samplesD2(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            samplesD3(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        else
            values = data.(['V',num2str(ci)])';
            stds = data.(['sdV', num2str(ci)])';
            samplesD1(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            samplesD2(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            samplesD3(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        end
    end
elseif strcmp(mode, 'cutoff')
    error('The cutoff boundary algorithm has not been developped yet.');
end
D1 = sum(samplesD1, 1)*wp + Mp;
D2 = sum(samplesD2, 1)*wp + Mp;
D3 = sum(samplesD3, 1)*wp + Mp;
D = [D1; D2; D3];
if gpuparallel
    SVs = samples./D + gpuArray.randn(size(samples))*eta;
    %choice = gpuArray(data.chosenItem');
else
    SVs = samples./D + randn(size(samples))*eta;
    %choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 1);
probs = squeeze(sum(max_from_each_distribution, 2) / size(SVs, 2));
end