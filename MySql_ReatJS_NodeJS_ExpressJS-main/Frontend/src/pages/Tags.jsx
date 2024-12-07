import React, { useState } from 'react';
import './FileUploadStyle.css';

function Tags() {
  const [selectedLeague, setSelectedLeague] = useState(null);

  const handleLeagueChange = (league) => {
    setSelectedLeague(league);
  };

  return (
    <div>
      <h1>Choose a League</h1>
      <div>
        <input
          type="radio"
          id="nfl"
          name="league"
          value="nfl"
          checked={selectedLeague === 'nfl'}
          onChange={() => handleLeagueChange('nfl')}
        />
        <label htmlFor="nfl">NFL</label>
        <input
          type="radio"
          id="mlb"
          name="league"
          value="mlb"
          checked={selectedLeague === 'mlb'}
          onChange={() => handleLeagueChange('mlb')}
        />
        <label htmlFor="mlb">MLB</label>
      </div>

      <div className='display_container'>
        {selectedLeague === 'nfl' && (
          <img
            src="../components/images/nfl_tag.png"
            alt="NFL"
            style={{ width: '200px', height: '200px' }}
          />
        )}
        {selectedLeague === 'mlb' && (
          <img
            src="../components/images/mlb_tag.png"
            alt="MLB"
            style={{ width: '200px', height: '200px' }}
          />
        )}
      </div>
    </div>
  );
}

export default Tags;
