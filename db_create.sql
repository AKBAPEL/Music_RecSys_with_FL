CREATE TABLE members (
  Member_ID VARCHAR(100) UNIQUE NOT NULL,
  city INTEGER,
  age INTEGER,
  gender VARCHAR(100),
  registred_via INTEGER,
  registration_init_time VARCHAR(100),
  expiration_date VARCHAR(100)
);

CREATE TABLE songs (
  Song_ID  VARCHAR(100) UNIQUE NOT NULL,
  song_length INTEGER,
  genre_ids INTEGER,
  artist_name VARCHAR(100),
  composer VARCHAR(100),
  lyricist VARCHAR(100),
  language INTEGER
);


CREATE TABLE song_extra_info (
  Song_ID  VARCHAR(100) UNIQUE NOT NULL,
  song_name VARCHAR(100),
  isrc VARCHAR(100)
);

CREATE TABLE event_data (
  Member_ID VARCHAR(100) UNIQUE NOT NULL,
  Song_ID  VARCHAR(100) UNIQUE NOT NULL,
  source_system_tab VARCHAR(100),
  source_screen_name VARCHAR(100),
  source_type VARCHAR(100),
  target INTEGER 
);