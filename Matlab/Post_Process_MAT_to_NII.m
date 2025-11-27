% % ==========================================================
% % Post_Process_MAT_to_NII.m (全新脚本)
% % ==========================================================
%
% 职责：
% 1. (新增) 定义dB阈值。
% 2. 查找 '01_Recon_MAT' 文件夹中的所有 .mat 文件。
% 3. (新增) 循环遍历所有 .mat 文件和所有 dB 阈值。
% 4. (新增) 在 '02_Post_Processed' 中创建有序的 NIfTI 和 JPG 文件夹。
% 5. 保存所有 .nii 和 .jpg 文件。
%
clc;
clear;
close all;

% --- 1. (!! 核心 !!) 定义处理参数 ---
MasterOutputDir = 'Ultrasound_Simulation_Data_500'; % (与 V6 脚本中一致)
db_levels = -60:5:-30; % (-60, -55, -50, -45, -40, -35, -30)
recon_types = {'Full_33_Angle', 'Zero_0_Degree'};
jpg_resolution = 300; % (JPG 文件的分辨率, e.g., 300)

fprintf('--- 开始后处理 (MAT -> NII/JPG) ---\n');
fprintf('--- 目标dB级别: %s ---\n', num2str(db_levels));
fprintf('--- 目标文件夹: %s ---\n', MasterOutputDir);

% --- 2. 定义输入/输出文件夹 ---
inputBaseDir = fullfile(MasterOutputDir, '03_Recon_MAT');
outputBaseDir = fullfile(MasterOutputDir, '05_Post_Processed');

if ~exist(outputBaseDir, 'dir'), mkdir(outputBaseDir); end

% --- 3. 循环处理所有重建类型 ('Full' 和 'Zero') ---
for type_cell = recon_types
    current_type = type_cell{1};
    fprintf('\n========================================\n');
    fprintf('正在处理类型: %s\n', current_type);
    
    current_inputDir = fullfile(inputBaseDir, current_type);
    current_output_nii_Base = fullfile(outputBaseDir, current_type, 'NII');
    current_output_jpg_Base = fullfile(outputBaseDir, current_type, 'JPG');
    
    if ~exist(current_inputDir, 'dir')
        fprintf(' > 警告: 找不到输入文件夹 %s. 跳过此类型。\n', current_inputDir);
        continue;
    end
    
    fileList = dir(fullfile(current_inputDir, 'Recon_Combined_*.mat'));
    if isempty(fileList)
        fprintf(' > 未在 %s 中找到 .mat 文件。\n', current_inputDir);
        continue;
    end
    
    fprintf(' > 找到 %d 个 .mat 文件。\n', length(fileList));

    % --- 4. 循环处理该类型中的所有 .mat 文件 ---
    for file_idx = 1:length(fileList)
        file = fileList(file_idx);
        matFilePath = fullfile(file.folder, file.name);
        [~, baseName, ~] = fileparts(file.name);
        
        fprintf('\n > 正在加载: %s (%d / %d)\n', file.name, file_idx, length(fileList));
        load(matFilePath, 'volume_combined', 'x_space_cropped', 'y_space_cropped', 'z_space_cropped');
        
        % --- 5. (!! 核心 !!) 循环遍历所有 dB 阈值 ---
        for db_level = db_levels
            fprintf('   >> 正在处理 %.0fdB...\n', db_level);
            
            % --- 5.1 创建动态输出文件夹 ---
            nii_db_folder = fullfile(current_output_nii_Base, sprintf('NII_%.0fdB', db_level));
            jpg_db_folder = fullfile(current_output_jpg_Base, sprintf('JPG_%.0fdB', db_level));
            if ~exist(nii_db_folder, 'dir'), mkdir(nii_db_folder); end
            if ~exist(jpg_db_folder, 'dir'), mkdir(jpg_db_folder); end
            
            % --- 5.2 保存 NIfTI (.nii) 文件 ---
            nii_filename = fullfile(nii_db_folder, [baseName, '.nii']);
            try
                voxel_spacing = [
                    (z_space_cropped(2) - z_space_cropped(1)), ...
                    (x_space_cropped(2) - x_space_cropped(1)), ...
                    (y_space_cropped(2) - y_space_cropped(1))
                ] * 1000;
                
                % (调用局部函数)
                volume_data_nii = process_nii_image_local(volume_combined, db_level); 
                
                niftiwrite(volume_data_nii, nii_filename, 'Compressed', false); 
                info = niftiinfo(nii_filename); 
                info.PixelDimensions = voxel_spacing;
                info.SpaceUnits = 'Millimeter';
                info.Datatype = 'single';
                niftiwrite(volume_data_nii, nii_filename, info, 'Compressed', false); 
                
            catch ME_save_nii
                warning(ME_save_nii.identifier, '保存 .nii (%.0fdB) 文件失败: %s', db_level, ME_save_nii.message);
            end
            
            % --- 5.3 保存预览图像 (.jpg) ---
            jpg_filename = fullfile(jpg_db_folder, [baseName, '.jpg']);
            try
                hFig = figure('Visible', 'off');
                set(hFig, 'Position', [100 100 1200 400]);
                
                % XY 平面 (在 20mm 深度)
                subplot(1,3,1);
                [~, z_index] = min(abs(z_space_cropped - 0.020));
                img_xy = process_image_local(squeeze(volume_combined(z_index,:,:)), db_level);
                imagesc(x_space_cropped*1000, y_space_cropped*1000, img_xy.');
                xlabel('Lateral Dist[mm]'); ylabel('Elevation Dist[mm]');
                colormap gray; colorbar; axis image; caxis([db_level 0]);
                
                % YZ 平面 (在 X=0 位置)
                subplot(1,3,2);
                [~, x_index] = min(abs(0 - x_space_cropped));
                img_yz = process_image_local(squeeze(volume_combined(:,x_index,:)), db_level);
                imagesc(y_space_cropped*1000, z_space_cropped*1000, img_yz);
                xlabel('Lateral Dist[mm]'); ylabel('Axial Dist[mm]');
                colormap gray; colorbar;
                xlim([-12.7 12.7]); ylim([z_space_cropped(1)*1000 z_space_cropped(end)*1000]);
                
                % XZ 平面 (在 Y=0 位置)
                subplot(1,3,3);
                [~, y_index] = min(abs(0 - y_space_cropped));
                img_xz = process_image_local(squeeze(volume_combined(:,:,y_index)), db_level);
                imagesc(x_space_cropped*1000, z_space_cropped*1000, img_xz);
                xlabel('Elevation Dist[mm]'); ylabel('Axial Dist[mm]');
                colormap gray; colorbar;
                xlim([-12.7 12.7]); ylim([z_space_cropped(1)*1000 z_space_cropped(end)*1000]);
                
                set(findall(hFig,'type','colorbar'),'Limits',[db_level 0]);
                print(hFig, jpg_filename, '-djpeg', sprintf('-r%d', jpg_resolution));
                close(hFig);
                
            catch ME_save_img
                warning(ME_save_img.identifier, '保存 .jpg 预览图 (%.0fdB) 失败: %s', db_level, ME_save_img.message);
                if exist('hFig', 'var') && ishandle(hFig)
                    close(hFig);
                end
            end
            
        end % 结束 dB 循环
        
        clear volume_combined x_space_cropped y_space_cropped z_space_cropped;
        
    end % 结束 file 循环
end % 结束 recon_type 循环

fprintf('\n========================================\n');
fprintf('--- 自动化后处理全部完成！ ---\n');


% % ==========================================================
% % 局部函数 (Local Functions)
% % ==========================================================
function img_envelope = process_image_local(img, db_level)
    % (接收 db_level 作为参数)
    img_envelope = abs(img);
    max_val = max(img_envelope(:));
    if max_val == 0, max_val = 1; end
    img_envelope = 20*log10(img_envelope./max_val);
    img_envelope(img_envelope < db_level) = db_level; % (应用动态dB)
end

function volume_data_nii = process_nii_image_local(volume_complex, db_level)
    % (接收 db_level 作为参数)
    volume_data_nii = abs(volume_complex);
    max_val = max(volume_data_nii(:));
    if max_val == 0
        max_val = 1; % 避免 log10(0)
    end
    volume_data_nii = 20*log10(volume_data_nii./max_val);
    volume_data_nii(volume_data_nii < db_level) = db_level; % (应用动态dB)
    volume_data_nii = single(volume_data_nii);
end