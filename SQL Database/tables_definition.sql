-- Drop dependent tables first
IF EXISTS (SELECT * FROM sys.tables WHERE name = 'NexusMuscleLink')
BEGIN
    DROP TABLE NexusMuscleLink;
END

IF EXISTS (SELECT * FROM sys.tables WHERE name = 'TracksMetrics')
BEGIN
    DROP TABLE TracksMetrics;
END

IF EXISTS (SELECT * FROM sys.tables WHERE name = 'Muscles')
BEGIN
    DROP TABLE Muscles;
END

IF EXISTS (SELECT * FROM sys.tables WHERE name = 'FunctionalFascialLines')
BEGIN
    DROP TABLE FunctionalFascialLines;
END

-- Now drop the referenced tables
IF EXISTS (SELECT * FROM sys.tables WHERE name = 'NexusNetwork')
BEGIN
    DROP TABLE NexusNetwork;
END

IF EXISTS (SELECT * FROM sys.tables WHERE name = 'Methods')
BEGIN
    DROP TABLE Methods;
END

IF EXISTS (SELECT * FROM sys.tables WHERE name = 'Stations')
BEGIN
    DROP TABLE Stations;
END


/* Creating Tables */
CREATE TABLE NexusNetwork (
    NexusID INT PRIMARY KEY,
    NexusName NVARCHAR(MAX) NOT NULL,
    Description NVARCHAR(MAX)
);

CREATE TABLE FunctionalFascialLines (
    LineID INT PRIMARY KEY,
	NexusID INT,
    LineName NVARCHAR(MAX) NOT NULL,
    FunctionDescription NVARCHAR(MAX),
	FOREIGN KEY (NexusID) REFERENCES NexusNetwork(NexusID),
);

CREATE TABLE Stations (
    StationID INT PRIMARY KEY,
    StationName NVARCHAR(MAX) NOT NULL
);

CREATE TABLE Muscles (
    MuscleID INT IDENTITY(1,1) PRIMARY KEY,
    MuscleName NVARCHAR(MAX) NOT NULL,
    Description NVARCHAR(MAX),
    UpperStationID INT,
    LowerStationID INT
);



CREATE TABLE NexusMuscleLink (
    NexusID INT,
    MuscleID INT,
    Type VARCHAR(12) NOT NULL CHECK (Type IN ('Compression', 'Activation')),
    OppositeSide TINYINT NOT NULL CHECK (OppositeSide IN (0, 1)),
);


CREATE TABLE TracksMetrics (
    MetricID INT PRIMARY KEY,
    LineID INT,
    Alpha FLOAT,
    Lambda FLOAT,
    Side NVARCHAR(255),
    FOREIGN KEY (LineID) REFERENCES FunctionalFascialLines(LineID)
);


CREATE TABLE Methods (
    MethodID INT IDENTITY(1,1) PRIMARY KEY,
    MethodName NVARCHAR(255) NOT NULL,
    Description NVARCHAR(MAX),
    MethodType NVARCHAR(50) NOT NULL CHECK (MethodType IN ('Training & Activating', 'Relaxing & Releasing')),
    TargetMuscleType NVARCHAR(255) NOT NULL,
    SuitableForHome BIT NOT NULL DEFAULT 1 -- 1 means suitable for home, 0 otherwise
);
