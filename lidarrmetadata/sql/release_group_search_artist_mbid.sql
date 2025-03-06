WITH release_data AS (
  SELECT
    release_group.gid AS gid,
    COALESCE(release_group_primary_type.name, 'Other') as primary_type,
    release_group.name AS album,
    (
      SELECT json_build_object(
        'id', r.gid,
        'date', r.release_date
      )
      FROM (
        SELECT DISTINCT ON (release_group.id) r.id,
          r.gid,
          COALESCE(
            MIN(make_date(rc.date_year, rc.date_month, rc.date_day)),
            MIN(make_date(COALESCE(rc.date_year, 1), COALESCE(rc.date_month, 1), COALESCE(rc.date_day, 1)))
          ) as release_date
        FROM release r
        LEFT JOIN LATERAL (
          SELECT date_year, date_month, date_day
          FROM release_country
          WHERE release_country.release = r.id
          UNION ALL
          SELECT date_year, date_month, date_day
          FROM release_unknown_country
          WHERE release_unknown_country.release = r.id
        ) rc ON true
        WHERE r.release_group = release_group.id
        GROUP BY release_group.id, r.id, r.gid
        ORDER BY release_group.id,
                 MIN(rc.date_year) DESC NULLS LAST,
                 MIN(rc.date_month) DESC NULLS LAST,
                 MIN(rc.date_day) DESC NULLS LAST,
                 r.id DESC
      ) r
    ) as primary_release
  FROM release_group
    JOIN artist_credit_name ON artist_credit_name.artist_credit = release_group.artist_credit
    JOIN artist ON artist_credit_name.artist = artist.id
    LEFT JOIN release_group_primary_type ON release_group.type = release_group_primary_type.id
  WHERE artist.gid = $1 AND artist_credit_name.position = 0
)
SELECT json_build_object(
  'Album', (
    SELECT json_agg(json_build_object(
      'id', gid,
      'title', album,
      'primary_release', primary_release
    ) ORDER BY album)
    FROM release_data
    WHERE primary_type = 'Album'
  ),
  'Single', (
    SELECT json_agg(json_build_object(
      'id', gid,
      'title', album,
      'primary_release', primary_release
    ) ORDER BY album)
    FROM release_data
    WHERE primary_type = 'Single'
  ),
  'EP', (
    SELECT json_agg(json_build_object(
      'id', gid,
      'title', album,
      'primary_release', primary_release
    ) ORDER BY album)
    FROM release_data
    WHERE primary_type = 'EP'
  ),
  'Broadcast', (
    SELECT json_agg(json_build_object(
      'id', gid,
      'title', album,
      'primary_release', primary_release
    ) ORDER BY album)
    FROM release_data
    WHERE primary_type = 'Broadcast'
  ),
  'Other', (
    SELECT json_agg(json_build_object(
      'id', gid,
      'title', album,
      'primary_release', primary_release
    ) ORDER BY album)
    FROM release_data
    WHERE primary_type = 'Other'
  )
) as result
