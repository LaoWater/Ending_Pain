   SELECT nn.NexusName, m.MuscleName,nml.Type, nml.OppositeSide
        FROM NexusMuscleLink nml
        INNER JOIN Muscles m ON m.MuscleID = nml.MuscleID
        INNER JOIN NexusNetwork nn ON nn.NexusID = nml.NexusID
        WHERE 1=1
        -- and nml.Type = 'Compression'
        -- and nn.NexusName = 'Origin Nexus'
        ORDER BY nn.NexusID, nml.OppositeSide, nml.Type DESC, m.MuscleID