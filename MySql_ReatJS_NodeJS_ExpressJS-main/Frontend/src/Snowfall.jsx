import React, { useEffect, useState } from 'react';
import styles from './Snowfall.module.scss';

const Snowfall = () => {
  // Create an array to store snowflakes
  const [flakes, setFlakes] = useState([]);

  // Generate random snowflakes
  useEffect(() => {
    const numberOfFlakes = 200; // Number of snowflakes to generate
    let flakesArray = [];

    // Generate random snowflakes with random positions, animation timings, etc.
    for (let i = 0; i < numberOfFlakes; i++) {
      flakesArray.push({
        id: i,
        animationDelay: `${Math.random() * 30}s`, // Random animation delay
        animationDuration: `${Math.random() * 20 + 10}s`, // Random animation duration
        leftPosition: `${Math.random() * 100}vw`, // Random horizontal position
        size: `${Math.random() * 5 + 5}px`, // Random size for the snowflakes
      });
    }

    setFlakes(flakesArray);
  }, []);

  return (
    <div className={styles.snowfall}>
      {flakes.map((flake) => (
        <div
          key={flake.id}
          className={styles.snow}
          style={{
            left: flake.leftPosition,
            width: flake.size,
            height: flake.size,
            animationDelay: flake.animationDelay,
            animationDuration: flake.animationDuration,
          }}
        />
      ))}
    </div>
  );
};

export default Snowfall;
