SELECT
  row_to_json(artist_data) artist
  FROM (
    SELECT
      artist.gid AS id,
      artist.name as artist_name,
      json_build_object(
        'begin', CASE 
          WHEN artist.begin_date_year IS NOT NULL 
          THEN make_date(
            artist.begin_date_year, 
            COALESCE(artist.begin_date_month, 1), 
            COALESCE(artist.begin_date_day, 1)
          )
          ELSE NULL 
        END,
        'end', CASE 
          WHEN artist.end_date_year IS NOT NULL 
          THEN make_date(
            artist.end_date_year, 
            COALESCE(artist.end_date_month, 12), 
            COALESCE(artist.end_date_day, 31)
          )
          ELSE NULL 
        END
      ) AS life_span,
      array(
        SELECT url.url
          FROM url
                 JOIN l_artist_url ON l_artist_url.entity0 = artist.id AND l_artist_url.entity1 = url.id
      ) AS links,
      array(
        SELECT INITCAP(genre.name)
          FROM genre
                 JOIN tag ON genre.name = tag.name
                 JOIN artist_tag ON artist_tag.tag = tag.id
         WHERE artist_tag.artist = artist.id
           AND artist_tag.count > 0
      ) AS genres
      FROM artist
     WHERE artist.gid = ANY($1::uuid[])
  ) artist_data
