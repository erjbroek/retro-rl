def format_time_from_seconds(seconds):
  seconds = int(seconds)
  if seconds < 60:
    return f"{seconds:02d}"
  elif seconds < 3600:
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"
  else:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"