%% LabX_GaussianDenoising.m
% Model Gaussian noise, compare filters, compute MSE/PSNR/SSIM
% Student: Mahfuzur Rahaman
% Course: Mathematical Algorithms (DSP) — Image Processing Labs

close all; clear; clc;

%% --- Setup output folder ---
outputDir = fullfile(pwd, 'figures_labX');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% --- 0) Load image (use 'original.jpg' if present, else cameraman.tif) ---
if exist('original.jpg','file')
    orig = im2double(imread('original.jpg'));
    if size(orig,3)==3
        origGray = rgb2gray(orig);
    else
        origGray = orig;
    end
    fprintf('Using original.jpg\n');
else
    % fallback builtin demo image
    origGray = im2double(imread('cameraman.tif'));
    fprintf('Using cameraman.tif (fallback)\n');
end
imwrite(im2uint8(origGray), fullfile(outputDir,'orig_gray.png'));

%% --- 1) Add Gaussian noise ---
noiseSigma = 0.04;              % standard deviation
noisy = imnoise(origGray, 'gaussian', 0, noiseSigma^2);
imwrite(im2uint8(noisy), fullfile(outputDir,'noisy.png'));

%% --- 2) Traditional filter: Wiener (adaptive) ---
wienerWindow = [5 5];
traditionalFiltered = wiener2(noisy, wienerWindow, noiseSigma^2);
imwrite(im2uint8(traditionalFiltered), fullfile(outputDir,'wiener_filtered.png'));

%% --- 3) AI filter (DnCNN) if available; else use a stronger BM3D-like fallback (if toolbox missing) ---
aiAvailable = false;
try
    if exist('denoisingNetwork','file') == 2
        net = denoisingNetwork('DnCNN'); % requires Deep Learning Toolbox & pretrained models
        aiFiltered = denoiseImage(noisy, net);
        aiAvailable = true;
        fprintf('DnCNN loaded and applied.\n');
    end
catch ME
    warning('DnCNN not available or failed: %s', ME.message);
    aiAvailable = false;
end

% If DnCNN not available, try imnlmfilt (non-local means) as strong fallback (Image Processing Toolbox)
if ~aiAvailable
    if exist('imnlmfilt','file') == 2
        aiFiltered = imnlmfilt(noisy); % good denoising fallback
        fprintf('Used non-local means (imnlmfilt) as AI fallback.\n');
    else
        % fallback: stronger Gaussian blur (not AI-quality, but will show difference)
        aiFiltered = imgaussfilt(noisy, 0.9);
        fprintf('Used Gaussian blur fallback (no advanced denoiser available).\n');
    end
end
imwrite(im2uint8(aiFiltered), fullfile(outputDir,'ai_filtered.png'));

%% --- 4) (Optional) Also show simple mean and median filters for comparison ---
meanFiltered = imfilter(noisy, fspecial('average',3), 'replicate');
medianFiltered = medfilt2(noisy, [3 3]);
imwrite(im2uint8(meanFiltered), fullfile(outputDir,'mean_filtered.png'));
imwrite(im2uint8(medianFiltered), fullfile(outputDir,'median_filtered.png'));

%% --- 5) Quantitative metrics: MSE, PSNR, SSIM ---
% Helper functions
MSE = @(A,B) mean((A(:)-B(:)).^2);
PSNR_dB = @(A,B) psnr(A,B);       % built-in (works for double in [0,1])
SSIM_val = @(A,B) ssim(A,B);      % built-in

% Compute metrics
mseNoisy = MSE(noisy, origGray);
psnrNoisy = PSNR_dB(noisy, origGray);
ssimNoisy = SSIM_val(noisy, origGray);

mseWiener = MSE(traditionalFiltered, origGray);
psnrWiener = PSNR_dB(traditionalFiltered, origGray);
ssimWiener = SSIM_val(traditionalFiltered, origGray);

mseAI = MSE(aiFiltered, origGray);
psnrAI = PSNR_dB(aiFiltered, origGray);
ssimAI = SSIM_val(aiFiltered, origGray);

mseMean = MSE(meanFiltered, origGray);
psnrMean = PSNR_dB(meanFiltered, origGray);
ssimMean = SSIM_val(meanFiltered, origGray);

mseMed = MSE(medianFiltered, origGray);
psnrMed = PSNR_dB(medianFiltered, origGray);
ssimMed = SSIM_val(medianFiltered, origGray);

%% --- 6) Print metrics ---
fprintf('\n--- Image Quality Metrics ---\n');
fprintf('Method\t\tMSE\t\tPSNR(dB)\tSSIM\n');
fprintf('Noisy\t\t%.6f\t%.4f\t\t%.4f\n', mseNoisy, psnrNoisy, ssimNoisy);
fprintf('Wiener\t\t%.6f\t%.4f\t\t%.4f\n', mseWiener, psnrWiener, ssimWiener);
fprintf('AI\t\t%.6f\t%.4f\t\t%.4f\n', mseAI, psnrAI, ssimAI);
fprintf('Mean(3x3)\t%.6f\t%.4f\t\t%.4f\n', mseMean, psnrMean, ssimMean);
fprintf('Median(3x3)\t%.6f\t%.4f\t\t%.4f\n', mseMed, psnrMed, ssimMed);

%% --- 7) Visualize & save a combined figure ---
figure('Position',[100 100 1300 300]);
subplot(1,5,1); imshow(origGray); title('Original');
subplot(1,5,2); imshow(noisy); title(sprintf('Noisy (\\sigma=%.3f)', noiseSigma));
subplot(1,5,3); imshow(traditionalFiltered); title('Wiener');
subplot(1,5,4); imshow(aiFiltered); title('AI / Fallback');
subplot(1,5,5); imshow(medianFiltered); title('Median');

saveas(gcf, fullfile(outputDir,'comparison_montage.png'));

%% --- 8) Save a small report file with numeric results ---
reportPath = fullfile(outputDir,'metrics_report.txt');
fid = fopen(reportPath,'w');
fprintf(fid,'Lab: Gaussian Noise Denoising Comparison\n\n');
fprintf(fid,'Noise sigma = %.6f\n\n', noiseSigma);
fprintf(fid,'Method, MSE, PSNR(dB), SSIM\n');
fprintf(fid,'Noisy, %.6f, %.4f, %.4f\n', mseNoisy, psnrNoisy, ssimNoisy);
fprintf(fid,'Wiener, %.6f, %.4f, %.4f\n', mseWiener, psnrWiener, ssimWiener);
fprintf(fid,'AI, %.6f, %.4f, %.4f\n', mseAI, psnrAI, ssimAI);
fprintf(fid,'Mean3x3, %.6f, %.4f, %.4f\n', mseMean, psnrMean, ssimMean);
fprintf(fid,'Median3x3, %.6f, %.4f, %.4f\n', mseMed, psnrMed, ssimMed);
fclose(fid);

fprintf('Results saved to folder: %s\n', outputDir);
