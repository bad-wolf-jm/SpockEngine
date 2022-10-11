function [dout, file_info] = static_noise(data, ch, cfg)
    % =============================================================================
    %  Copyright (c) LeddarTech Inc. All rights reserved.
    % =============================================================================
    %  PROPRIETARY DATA NOTICE
    %  No part of this file shall be used, communicated, reproduced
    %  or copied in any form or by any means without the prior written
    %  permission of LeddarTech Inc.
    % =============================================================================
    % Static Noise Removal
    %
    % [dout, file_info] = static_noise(data, ch, cfg)
    %
    % data : Trace data samples
    % ch : Current channel
    % cfg  : configuration structure
    %
    % dout : Corrected data out
    % file_info: File information for results writing

    header_size = 16;
    set_lentgh = 32;
    template_length = 64;
    ch += 1;

    % convert to column vector
    sz = size(data);

    if sz(2) > sz(1)
        d = data';
    else
        d = data;
    endif

    if(~cfg.block_enable)
        dout = d;
    else

        bfile = fopen(cfg.template);
        bdata = fread(bfile, 'short');

        start_index = cfg.template_offset(ch) + cfg.global_offset + 1;
        end_index = start_index + template_length - 1;

        mem_start_offset = header_size + (cfg.set_sel * template_length * set_lentgh) + (template_length * cfg.pd_mapping(ch)) + start_index;
        mem_end_offset = mem_start_offset + template_length - 1;

        template = zeros(numel(d), 1);
        template(start_index:end_index) = bdata(mem_start_offset:mem_end_offset);

        dout = d - template;
        fclose(bfile);
    endif;

    file_info.header = {'trace'; 'data_o'; 'last'};
    file_info.data = [dout];

endfunction

% vim:tw=0:ts=2:sts=2:sw=2:et
