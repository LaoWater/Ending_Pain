INSERT INTO NexusNetwork (NexusID, NexusName, Description) VALUES
(1, 'Origin Nexus', 'Connects body to ground'),
(2, 'Nebula Nexus', 'Center of kinetic energy Generation & Absorption'),
(3, 'Horizon Nexus', 'Functioning of extremities');


INSERT INTO FunctionalFascialLines (LineID, NexusID, LineName, FunctionDescription) VALUES
(1, 1, 'Lateral Line', 'Facilitates lateral movement and stabilization, spanning from the foot to the head along the body’s sides.'),
(2, 2, 'Oblique Sling', 'Supports rotational forces and transfers energy between the upper and lower body.'),
(3, 2, 'Lateral Line Proximal', 'Handles upper body lateral stability and movement, particularly in the torso.'),
(4, 1, 'Lateral Line Inferior', 'Manages lower body lateral movements and contributes to lower limb stability.'),
(5, 3, 'Lateral Line Superior', 'Aids in stability and movement of the head and neck in lateral directions.'),
(6, 3, 'Arm Line', 'Controls the movements and stabilization of the arm, facilitating various arm motions.');


INSERT INTO Stations (StationID, StationName) VALUES
(0, 'Nose'),
(9, 'Mouth left'),
(10, 'Mouth right'),
(11, 'Left shoulder'),
(12, 'Right shoulder'),
(14, 'Right elbow'),
(15, 'Left wrist'), 
(16, 'Right wrist'),
(17, 'Left pinky'),
(18, 'Right pinky'),
(19, 'Left index'),
(20, 'Right index'),
(21, 'Left thumb'),
(22, 'Right thumb'),
(23, 'Left hip'),
(24, 'Right hip'),
(25, 'Left knee'),
(26, 'Right knee'),
(27, 'Left ankle'),
(28, 'Right ankle'),
(29, 'Left heel'),
(30, 'Right heel'),
(31, 'Left foot index'),
(32, 'Right foot index');


INSERT INTO Muscles (MuscleName, Description, LowerStationID, UpperStationID) VALUES
-- Lower Body Muscles
('Exterior Plantar Fascia', 'Fibrous tissue that supports the arch on the outer side of the bottom of the foot', 32, 30),
('Interior Plantar Fascia', 'Fibrous tissue that supports the arch on the inner side of the bottom of the foot', 31, 29),
('Gastrocnemius', 'Calf muscle that flexes the knee and bends the foot downward', 27, 25),
('Soleus', 'Muscle that runs from just below the knee to the heel, important for standing and walking', 29, 27),
('Tibialis Anterior', 'Muscle that spans the length of the tibia and enables dorsiflexion and inversion of the ankle', 27, 25),
('Quadriceps Femoris', 'Group of muscles located in front of the thigh, responsible for knee extension', 25, 23),
('Hamstrings', 'Muscle group at the back of the thigh, involved in knee flexion', 25, 23),
('IT Band', 'Fibrous tissue that extends from the hip to the outside of the thigh and stabilizes the knee', 23, 25),
('Gluteus Maximus', 'Large muscle of the buttocks that extends the hip', 23, 0),
('Gluteus Medius', 'Muscle on the outer side of the buttocks that abducts and rotates the thigh', 23, 0),
('Adductor Magnus', 'Large thigh muscle that adducts and flexes the thigh', 25, 23),

-- Core and Upper Body Muscles
('Rectus Abdominis', 'Known as the abs muscle, runs along the front of the abdomen', 23, 11),
('External Oblique', 'Located on the side and front of the abdomen', 23, 11),
('Internal Oblique', 'Located on the side and front of the abdomen', 23, 11),
('Quadratus Lumborum', 'Muscle of the lower back that stabilizes the lumbar spine and aids in lateral movement', 23, 11),
('Erector Spinae', 'Muscle that runs along the spine, maintains posture', 23, 11),
('Pectoralis Major', 'Chest muscle that causes the arm to rotate inwardly', 11, 13),
('Pectoralis Minor', 'Small chest muscle that draws the scapula downward and forward', 11, 13),
('Latissimus Dorsi', 'Large tmuscle on the back that moves the arm', 11, 23),
('Deltoid', 'Shoulder muscle for arm rotation', 11, 13),
('Biceps Brachii', 'Front of the upper arm, flexes the elbow', 13, 15),
('Triceps Brachii', 'Back of the upper arm, extends the elbow', 13, 15),
('Forearm Flexors', 'Muscle group that flexes the fingers and wrist', 15, 17),
('Forearm Extensors', 'Muscle group that extends the fingers and wrist', 15, 19),
('Trapezius', 'The trapezius (traps muscle) helps you move your head, neck, arms, shoulders and torso.', 0, 11),
('Teres Minor', 'Teres minor is a rotator cuff muscle that externally rotates and adducts the arm', 15, 19),
('Teres Major', 'The TM muscle acts as a function as a unit with the latissimus dorsi (LD), where it acts in synergy to extend, adduct and internally rotate the shoulder.', 0, 11),

/* Finer Muslces */
('Levator Scapulae', 'Elevates the scapula and tilts its glenoid cavity inferiorly by rotating the scapula.', 11, 0), 
('Scalenes', 'Group of three pairs of muscles in the lateral neck, primarily involved in respiration and assisting in neck flexion.', 0, 0), 
('Masseter', 'Facial muscle that plays a major role in the chewing of solid foods, one of the strongest muscles in the human body.', 9, 10),
('SCM', 'Anchorer of head', 9, 10),
('Biceps', 'A large muscle that lies on the front of the upper arm between the shoulder and the elbow.', 11, 14); 


/* Origin Nexus */
INSERT INTO NexusMuscleLink (NexusID, MuscleID, Type, OppositeSide)
SELECT n.NexusID, m.MuscleID, 'Compression' AS Type, 0 AS OppositeSide
FROM NexusNetwork n
JOIN Muscles m ON n.NexusName = 'Origin Nexus' AND m.MuscleName IN (
    'Exterior Plantar Fascia',
    'IT Band',
    'Gluteus Medius',
    'Quadratus Lumborum',
    'External Oblique'
    -- *Optional - can deepen 
)

UNION ALL

SELECT n.NexusID, m.MuscleID, 'Activation' AS Type, 0 AS OppositeSide
FROM NexusNetwork n
JOIN Muscles m ON n.NexusName = 'Origin Nexus' AND m.MuscleName IN (
    'Interior Plantar Fascia',
    'Adductor Magnus' )

UNION ALL

SELECT n.NexusID, m.MuscleID, 'Activation' AS Type, 1 AS OppositeSide
FROM NexusNetwork n
JOIN Muscles m ON n.NexusName = 'Origin Nexus' AND m.MuscleName IN (
    'Exterior Plantar Fascia',
    'Tibialis Anterior',
	'IT Band',
	'External Oblique',
	'Quadratus Lumborum'
);



/*Nebula Nexus */
-- Inserting 'Compression' type for Nebula Nexus
INSERT INTO NexusMuscleLink (NexusID, MuscleID, Type, OppositeSide)

SELECT n.NexusID, m.MuscleID, 'Compression' AS Type, 0 AS OppositeSide
FROM NexusNetwork n
JOIN Muscles m ON n.NexusName = 'Nebula Nexus' AND m.MuscleName IN (
    'Gluteus Maximus',
    'Latissimus Dorsi',
    'Internal Oblique',
    'Pectoralis Major'
)

UNION ALL

-- Inserting 'Activation' type with no opposite side for Nebula Nexus
SELECT n.NexusID, m.MuscleID, 'Activation' AS Type, 0 AS OppositeSide
FROM NexusNetwork n
JOIN Muscles m ON n.NexusName = 'Nebula Nexus' AND m.MuscleName IN (
    'Adductor Magnus'
)

UNION ALL

-- Inserting 'Activation' type with opposite side for Nebula Nexus
SELECT n.NexusID, m.MuscleID, 'Activation' AS Type, 1 AS OppositeSide
FROM NexusNetwork n
JOIN Muscles m ON n.NexusName = 'Nebula Nexus' AND m.MuscleName IN (
    'Latissimus Dorsi',
    'Internal Oblique',
    'Quadratus Lumborum',
    'Gluteus Maximus'
);



/*Horizon Nexus */
-- Inserting 'Compression' type for Horizon Nexus without opposite side
INSERT INTO NexusMuscleLink (NexusID, MuscleID, Type, OppositeSide)
SELECT n.NexusID, m.MuscleID, 'Compression' AS Type, 0 AS OppositeSide
FROM NexusNetwork n
JOIN Muscles m ON n.NexusName = 'Horizon Nexus' AND m.MuscleName IN (
    'Trapezius',
    'Levator Scapulae',
    'Scalenes',
    'Masseter',
    'SCM',
    'Biceps',
    'Forearm Flexors'
)

UNION ALL

-- Inserting 'Activation' type for Horizon Nexus without opposite side
SELECT n.NexusID, m.MuscleID, 'Activation' AS Type, 0 AS OppositeSide
FROM NexusNetwork n
JOIN Muscles m ON n.NexusName = 'Horizon Nexus' AND m.MuscleName IN (
    'Forearm Extensors'
)

UNION ALL

-- Inserting 'Compression' type for Horizon Nexus with opposite side
SELECT n.NexusID, m.MuscleID, 'Compression' AS Type, 1 AS OppositeSide
FROM NexusNetwork n
JOIN Muscles m ON n.NexusName = 'Horizon Nexus' AND m.MuscleName IN (
    'Deltoid',
    'Forearm Extensors'
)

UNION ALL

-- Inserting 'Activation' type for Horizon Nexus with opposite side
SELECT n.NexusID, m.MuscleID, 'Activation' AS Type, 1 AS OppositeSide
FROM NexusNetwork n
JOIN Muscles m ON n.NexusName = 'Horizon Nexus' AND m.MuscleName IN (
    'SCM',
    'Scalenes'
);


/*

-- Muscles to Nexus/Lines connection
-- Linking muscles to the Origin Nexus (foundation and ground connection)
INSERT INTO NexusMuscleLink (NexusID, MuscleID)
SELECT n.NexusID, m.MuscleID
FROM NexusNetwork n, Muscles m
WHERE n.NexusName = 'Origin Nexus' AND m.MuscleName IN (
    'Exterior Plantar Fascia',
    'IT Band',
    'Gluteus Medius',
    'Quadratus Lumborum',
	'External Oblique'
    -- *Optional - can deepen 
);

-- Linking muscles to the Nebula Nexus (Core kinetic energy generation & absorption)
INSERT INTO NexusMuscleLink (NexusID, MuscleID)
SELECT n.NexusID, m.MuscleID
FROM NexusNetwork n, Muscles m
WHERE n.NexusName = 'Nebula Nexus' AND m.MuscleName IN (
    'Latissimus Dorsi',
	'Adductor Magnus',
	'Gluteus Maximus',
	'Pectoralis Major',
    'External Oblique'


);

-- Linking muscles to the Horizon Nexus (functioning of extremities)
INSERT INTO NexusMuscleLink (NexusID, MuscleID)
SELECT n.NexusID, m.MuscleID
FROM NexusNetwork n, Muscles m
WHERE n.NexusName = 'Horizon Nexus' AND m.MuscleName IN (
	'Trapezius',
    'Pectoralis Major',
    'Pectoralis Minor',
    'Deltoid',
    'Biceps Brachii',
    'Triceps Brachii'
    -- Add other muscles primarily associated with extremity movement and stabilization
);

*/


INSERT INTO Methods (MethodName, Description, MethodType, TargetMuscleType, SuitableForHome) VALUES
('Resistance Band Workouts', 'Use of resistance bands to perform various strength exercises at home', 'Training & Activating', 'General', 1),
('Body Weight Exercises', 'Exercises that use body weight as resistance, such as push-ups, squats, and lunges', 'Training & Activating', 'Full Body', 1),
('Guided Meditation', 'Audio or video-led sessions that guide individuals through relaxation techniques', 'Relaxing & Releasing', 'Mental & Physical Health', 1),
('Self-Myofascial Release', 'Techniques using foam rollers or massage balls to perform self-massage, releasing muscle tightness', 'Relaxing & Releasing', 'General', 1),
('Yoga', 'Home-based yoga sessions focusing on flexibility, strength, and relaxation', 'Training & Activating', 'General', 1),
('Breathwork Exercises', 'Techniques that involve controlling the breathing pattern to improve relaxation', 'Relaxing & Releasing', 'Respiratory System', 1),
('Pilates', 'Exercise system that improves flexibility, builds strength and develops control and endurance in the entire body', 'Training & Activating', 'Full Body', 1),
('Stretching Routines', 'Simple stretching exercises aimed to improve flexibility and relax muscles', 'Relaxing & Releasing', 'General', 1),
('Isometric Holds', 'Exercises that involve holding a position without movement to build strength', 'Training & Activating', 'Specific Muscle Groups', 1),
('Progressive Muscle Relaxation', 'Technique of tensing and then relaxing each muscle group in sequence', 'Relaxing & Releasing', 'General', 1);
