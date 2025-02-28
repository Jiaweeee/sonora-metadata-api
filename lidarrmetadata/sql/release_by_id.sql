WITH release_info AS (
  SELECT
    release.gid AS Id,
    array(
      SELECT gid
        FROM release_gid_redirect
       WHERE release_gid_redirect.new_id = release.id
    ) as OldIds,
    release.name AS Title,
    release.comment AS Disambiguation,
    release_status.name AS Status,
    -- 添加 artist 信息
    artist.gid as ArtistId,
    artist.name as ArtistName,
    -- 添加 cover art archive 信息
    array(
      SELECT json_build_object(
        'id', index_listing.id
      )
      FROM cover_art_archive.index_listing
      WHERE index_listing.release = release.id
      ORDER BY index_listing.id
    ) as images,
    -- 添加 link 信息
    array(
      SELECT json_build_object(
        'url', url.url,
        'type', link_type.name
      )
      FROM l_release_url
      JOIN url ON url.id = l_release_url.entity1
      JOIN link ON link.id = l_release_url.link
      JOIN link_type ON link_type.id = link.link_type
      WHERE l_release_url.entity0 = release.id
    ) as links,
    -- 原有的其他字段保持不变
    (
      SELECT 
        COALESCE(
          MIN(make_date(date_year, date_month, date_day)),
          MIN(make_date(COALESCE(date_year, 1), COALESCE(date_month, 1), COALESCE(date_day, 1)))
          )
        FROM (
          SELECT date_year, date_month, date_day
            FROM release_country
           WHERE release_country.release = release.id
                 
           UNION

          SELECT date_year, date_month, date_day
            FROM release_unknown_country
           WHERE release_unknown_country.release = release.id
        ) dates
    ) AS ReleaseDate,
    array(
      SELECT name FROM label
                     JOIN release_label ON release_label.label = label.id
       WHERE release_label.release = release.id
       ORDER BY name ASC
    ) AS Label,
    array(
      SELECT name FROM area
                     JOIN country_area ON country_area.area = area.id
                     JOIN release_country ON release_country.country = country_area.area
       WHERE release_country.release = release.id
    ) AS Country,
    array(
      SELECT json_build_object(
        'Format', medium_format.name,
        'Name', medium.name,
        'Position', medium.position
      ) FROM medium
               JOIN medium_format ON medium_format.id = medium.format
       WHERE medium.release = release.id
       ORDER BY medium.position
    ) AS Media,
    (SELECT SUM(medium.track_count) FROM medium WHERE medium.release = release.id) AS track_count,
    (
      SELECT
        COALESCE(json_agg(row_to_json(track_data)), '[]'::json)
        FROM (
          SELECT
            track.gid AS Id,
            array(
              SELECT gid
                FROM track_gid_redirect
               WHERE track_gid_redirect.new_id = track.id
            ) as OldIds,
            recording.gid AS RecordingId,
            array(
              SELECT gid
                FROM recording_gid_redirect
               WHERE recording_gid_redirect.new_id = recording.id
            ) as OldRecordingIds,
            artist.gid AS ArtistId,
            track.name AS TrackName,
            track.length AS DurationMs,
            medium.position AS MediumNumber,
            track.number AS TrackNumber,
            track.position AS TrackPosition
            FROM track
                   JOIN medium ON track.medium = medium.id
                   JOIN artist_credit_name ON artist_credit_name.artist_credit = track.artist_credit
                   JOIN artist ON artist_credit_name.artist = artist.id
                   JOIN recording ON track.recording = recording.id
           WHERE medium.release = release.id
             AND artist_credit_name.position = 0
             AND recording.video = FALSE
             AND track.is_data_track = FALSE
        ) track_data
    ) AS Tracks,
    json_build_object(
      'Id', release_group.gid,
      'OldIds', (
        SELECT array_agg(gid)
          FROM release_group_gid_redirect
         WHERE release_group_gid_redirect.new_id = release_group.id
      ),
      'Title', release_group.name,
      'Type', COALESCE(release_group_primary_type.name, 'Other'),
      'SecondaryTypes', (
        SELECT array_agg(name ORDER BY name)
          FROM release_group_secondary_type rgst
                 JOIN release_group_secondary_type_join rgstj ON rgstj.secondary_type = rgst.id
         WHERE rgstj.release_group = release_group.id
      ),
      'Disambiguation', release_group.comment,
      'Links', (
        SELECT array_agg(json_build_object(
          'target', url.url,
          'type', link_type.name
        ))
        FROM l_release_group_url
        JOIN url ON url.id = l_release_group_url.entity1
        JOIN link ON link.id = l_release_group_url.link
        JOIN link_type ON link_type.id = link.link_type
        WHERE l_release_group_url.entity0 = release_group.id
      ),
      'ReleaseDate', COALESCE(
        make_date(
          release_group_meta.first_release_date_year,
          release_group_meta.first_release_date_month,
          release_group_meta.first_release_date_day
        ),
        make_date(
          COALESCE(release_group_meta.first_release_date_year, 1),
          COALESCE(release_group_meta.first_release_date_month, 1),
          COALESCE(release_group_meta.first_release_date_day, 1)
        )
      )
    ) AS ReleaseGroup
  FROM release
         JOIN release_status ON release_status.id = release.status
         JOIN release_group ON release.release_group = release_group.id
         -- 添加艺术家相关的连接
         JOIN artist_credit ON release.artist_credit = artist_credit.id
         JOIN artist_credit_name ON artist_credit_name.artist_credit = artist_credit.id
         JOIN artist ON artist_credit_name.artist = artist.id
         -- 原有的其他连接保持不变
         LEFT JOIN release_group_meta ON release_group_meta.id = release_group.id
         LEFT JOIN release_group_primary_type ON release_group.type = release_group_primary_type.id
  WHERE release.gid = ANY($1::uuid[])
    AND artist_credit_name.position = 0  -- 获取主要艺术家
)
SELECT row_to_json(release_data) release
FROM release_info release_data;