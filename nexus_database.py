import json
import pyodbc


def process_origin_nexus_dataset(value):
    side_control_l = 'Left '
    side_control_r = 'Right '
    if value < 0:
        side_control_l = 'Right '
        side_control_r = 'Left '

    sql_query = f"""
    SELECT nn.NexusName,
    case when nml.OppositeSide = 0 then '{side_control_r}' else '{side_control_l}' end + m.MuscleName,
    case nml.Type when 'Compression' then 'Release/Lengthen' else 'Activate/Shorten' end
    FROM NexusMuscleLink nml
    INNER JOIN Muscles m ON m.MuscleID = nml.MuscleID
    INNER JOIN NexusNetwork nn ON nn.NexusID = nml.NexusID
    WHERE 1=1
    and nn.NexusName = 'Origin Nexus'
    ORDER BY nn.NexusID, nml.OppositeSide, nml.Type DESC, m.MuscleID
    """
    query_database(sql_query)
    json_results = query_database_json(sql_query)
    return json_results


def process_nebula_nexus_dataset(value):
    side_control_l = 'Left '
    side_control_r = 'Right '
    if value < 0:
        side_control_l = 'Right '
        side_control_r = 'Left '

    sql_query = f"""
    SELECT nn.NexusName,
    case when nml.OppositeSide = 0 then '{side_control_r}' else '{side_control_l}' end + m.MuscleName,
    case nml.Type when 'Compression' then 'Release/Lengthen' else 'Activate/Shorten' end
    FROM NexusMuscleLink nml
    INNER JOIN Muscles m ON m.MuscleID = nml.MuscleID
    INNER JOIN NexusNetwork nn ON nn.NexusID = nml.NexusID
    WHERE 1=1
    and nn.NexusName = 'Nebula Nexus'
    ORDER BY nn.NexusID, nml.OppositeSide, nml.Type DESC, m.MuscleID
    """
    query_database(sql_query)
    json_results = query_database_json(sql_query)
    return json_results


def process_horizon_nexus_dataset(value):
    side_control_l = 'Left '
    side_control_r = 'Right '
    if value < 0:
        side_control_l = 'Right '
        side_control_r = 'Left '

    sql_query = f"""
    SELECT nn.NexusName,
    case when nml.OppositeSide = 0 then '{side_control_r}' else '{side_control_l}' end + m.MuscleName,
    case nml.Type when 'Compression' then 'Release/Lengthen' else 'Activate/Shorten' end
    FROM NexusMuscleLink nml
    INNER JOIN Muscles m ON m.MuscleID = nml.MuscleID
    INNER JOIN NexusNetwork nn ON nn.NexusID = nml.NexusID
    WHERE 1=1
    and nn.NexusName = 'Horizon Nexus'
    ORDER BY nn.NexusID, nml.OppositeSide, nml.Type DESC, m.MuscleID
    """
    query_database(sql_query)
    json_results = query_database_json(sql_query)
    return json_results


def connect_to_database():
    # For Windows Authentication
    server = r'DESKTOP-48MNGTA\SQLEXPRESS'  # Your server name
    database = 'master'  # Your database name

    # Connection string for Windows Authentication
    conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'
    conn = pyodbc.connect(conn_str)

    return conn


def query_database(query):
    conn = connect_to_database()
    cursor = conn.cursor()

    # Execute the query
    cursor.execute(query)

    # Fetch and print each row
    for row in cursor.fetchall():
        print(row)

    print('\n')

    cursor.close()


def query_database_json(query):
    conn = connect_to_database()
    cursor = conn.cursor()

    # Execute the query
    cursor.execute(query)

    # Initialize an empty list to store the results
    results = []

    # Fetch each row and append to the list as a dictionary
    for row in cursor.fetchall():
        result = {
            "Nexus": row[0],
            "Target Muscle": row[1],
            "Action": row[2]
        }
        results.append(result)

    # Convert the list of dictionaries to a JSON string
    json_results = json.dumps(results, indent=4)  # 'indent' for pretty-printing

    # Close the connection
    cursor.close()
    conn.close()

    return json_results


####################
# Main Query Start #
####################
# origin_json_results = process_origin_nexus_dataset(0.2)
# nebula_json_results = process_nebula_nexus_dataset(0.33)
# horizon_json_results = process_horizon_nexus_dataset(-0.21)
